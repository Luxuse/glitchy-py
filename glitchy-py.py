import tkinter as tk
from tkinter import filedialog, messagebox, Scale, HORIZONTAL, ttk
from PIL import Image, ImageTk, ImageFilter
import numpy as np
import random
import os
import io
import sys # Pour vérifier la version de Python si besoin (ex: Pillow resampling)

# --- Constantes de configuration ---
MAX_DISPLAY_SIZE = (800, 600) # Taille maximale de l'image affichée dans la fenêtre
IMAGE_FILETYPES = [
    ("Fichiers image", "*.png *.jpg *.jpeg *.gif *.bmp"),
    ("Tous les fichiers", "*.*")
]
DEFAULT_SAVE_EXTENSION = ".png"

# Définir la méthode de resampling pour Pillow (change avec les versions)
# Utiliser Resampling.LANCZOS est généralement recommandé pour la qualité
try:
    # Pillow 9.1.0 et plus utilise Resampling enum
    Resampling = Image.Resampling.LANCZOS
except AttributeError:
    # Versions antérieures utilisent le filtre directement
    Resampling = Image.LANCZOS


# --- Fonctions utilitaires ---

def ensure_rgb_or_rgba(image_pil: Image.Image) -> Image.Image:
    """
    Assure que l'image est en mode RGBA. Convertit si nécessaire.
    RGBA est choisi pour gérer la transparence et la compatibilité.
    """
    if image_pil.mode != 'RGBA':
        # Convertir en RGBA. Le canal alpha sera ajouté si manquant (opaque par défaut).
        return image_pil.convert('RGBA')
    return image_pil

# --- Fonctions d'algorithmes de Glitch Destructeurs (avec paramètres) ---
# Chaque fonction prend une image PIL (attendre RGBA) et les paramètres nécessaires,
# et retourne une NOUVELLE image PIL glitchée (en mode RGBA).

def glitch_pixel_sort(image_pil: Image.Image, threshold=100, sort_by='luminosity', direction='horizontal') -> Image.Image:
    """
    Glitch de tri de pixels. Trie les pixels dans des segments basés sur un seuil.

    Args:
        image_pil: L'image PIL d'entrée (attendu en mode RGBA).
        threshold: Le seuil (0-255) pour déterminer les segments à trier.
        sort_by: Le critère de tri ('luminosity', 'red', 'green', 'blue').
        direction: La direction du tri ('horizontal' ou 'vertical').

    Returns:
        Une nouvelle image PIL glitchée (RGBA).
    """
    if not image_pil: return None
    # ensure_rgb_or_rgba est appelé par le wrapper avant d'appeler cette fonction
    img_np = np.array(image_pil) # RGBA -> (H, W, 4)
    height, width, channels = img_np.shape

    threshold = max(0, min(int(threshold), 255))

    # Base de tri - Ignorer le canal alpha pour la luminosité/couleur
    if sort_by == 'luminosity':
        base_sort = np.sum(img_np[:,:,:3].astype(np.float32) * [0.299, 0.587, 0.114], axis=2)
    elif sort_by == 'red': base_sort = img_np[:,:,0]
    elif sort_by == 'green': base_sort = img_np[:,:,1]
    elif sort_by == 'blue': base_sort = img_np[:,:,2]
    else: base_sort = np.sum(img_np[:,:,:3].astype(np.float32) * [0.299, 0.587, 0.114], axis=2) # Fallback

    img_glitched_np = img_np.copy() # Copie pour éviter de modifier l'original NP array directement

    # Appliquer le tri
    if direction == 'horizontal':
        for y in range(height):
            row_base = base_sort[y, :]
            mask = (row_base >= threshold).astype(int)
            diff_mask = np.diff(np.concatenate(([0], mask, [0])))
            starts = np.where(diff_mask == 1)[0]
            ends = np.where(diff_mask == -1)[0]
            intervals = list(zip(starts, ends))

            for start, end in intervals:
                if start < end:
                    # Trier le segment basé sur base_sort et appliquer l'ordre aux canaux RGBA
                    segment = img_glitched_np[y, start:end, :]
                    segment_base = row_base[start:end]
                    sorted_indices = np.argsort(segment_base)
                    img_glitched_np[y, start:end, :] = segment[sorted_indices]

    elif direction == 'vertical':
        for x in range(width):
            col_base = base_sort[:, x]
            mask = (col_base >= threshold).astype(int)
            diff_mask = np.diff(np.concatenate(([0], mask, [0])))
            starts = np.where(diff_mask == 1)[0]
            ends = np.where(diff_mask == -1)[0]
            intervals = list(zip(starts, ends))

            for start, end in intervals:
                 if start < end:
                    segment = img_glitched_np[start:end, x, :]
                    segment_base = col_base[start:end]
                    sorted_indices = np.argsort(segment_base)
                    img_glitched_np[start:end, x, :] = segment[sorted_indices]

    return Image.fromarray(img_glitched_np, image_pil.mode) # Retourne en RGBA


def glitch_byte_destruction(image_pil: Image.Image, num_bytes_to_modify=10000, destruction_type='zero', max_byte_change=100) -> Image.Image:
    """
    Glitch de destruction d'octets. Modifie ou échange des octets aléatoires dans les données de pixels.

    Args:
        image_pil: L'image PIL d'entrée (attendu en mode RGBA).
        num_bytes_to_modify: Le nombre d'octets à affecter.
        destruction_type: Le type de modification ('zero', 'random', 'invert', 'add_random', 'swap').
        max_byte_change: La variation maximale si destruction_type est 'add_random'.

    Returns:
        Une nouvelle image PIL glitchée (RGBA). Retourne l'original si échec critique.
    """
    if not image_pil: return None
    # ensure_rgb_or_rgba est appelé avant
    img = image_pil.copy() # Copie l'image PIL
    img_bytes = bytearray(img.tobytes()) # Convertit en bytes modifiables
    data_len = len(img_bytes)
    if data_len == 0: return img


    num_bytes_to_modify = max(0, min(int(num_bytes_to_modify), data_len))
    max_byte_change = max(0, min(int(max_byte_change), 255))


    # Sélectionner des positions aléatoires
    if num_bytes_to_modify > data_len // 2:
         all_positions = list(range(data_len))
         random.shuffle(all_positions)
         positions_to_modify = all_positions[:num_bytes_to_modify]
    else:
         try: # random.sample peut échouer si k > n (num_bytes_to_modify > data_len)
            positions_to_modify = random.sample(range(data_len), num_bytes_to_modify)
         except ValueError:
             positions_to_modify = [] # Pas assez de bytes, rien à modifier


    # Appliquer les modifications
    if destruction_type == 'zero':
        for pos in positions_to_modify:
            img_bytes[pos] = 0
    elif destruction_type == 'random':
         for pos in positions_to_modify:
            img_bytes[pos] = random.randint(0, 255)
    elif destruction_type == 'invert':
         for pos in positions_to_modify:
             img_bytes[pos] = 255 - img_bytes[pos]
    elif destruction_type == 'add_random':
         for pos in positions_to_modify:
            change = random.randint(-max_byte_change, max_byte_change)
            img_bytes[pos] = (img_bytes[pos] + change) % 256
    elif destruction_type == 'swap':
         num_swaps = len(positions_to_modify) // 2 # On utilise les positions sélectionnées
         if num_swaps > 0:
            # On choisit les paires parmi les positions déjà sélectionnées
            swap_indices = random.sample(range(len(positions_to_modify)), num_swaps * 2)
            for i in range(num_swaps):
                pos1 = positions_to_modify[swap_indices[2*i]]
                pos2 = positions_to_modify[swap_indices[2*i + 1]]
                img_bytes[pos1], img_bytes[pos2] = img_bytes[pos2], img_bytes[pos1]


    # Reconvertir en image PIL
    try: return Image.frombytes(img.mode, img.size, bytes(img_bytes))
    except ValueError:
        # Gérer l'échec si les bytes sont devenus totalement invalides
        print("Erreur: Les données corrompues ne peuvent pas être reconstruites par Image.frombytes.")
        messagebox.showwarning("Glitch Échoué", "La corruption de données a rendu l'image invalide.")
        return image_pil # Retourne l'original non corrompu


def glitch_channel_shuffle(image_pil: Image.Image) -> Image.Image:
    """
    Glitch de mélange de canaux. Mélange aléatoirement les canaux R, G, B.

    Args:
        image_pil: L'image PIL d'entrée (attendu en mode RGBA).

    Returns:
        Une nouvelle image PIL glitchée (RGBA).
    """
    if not image_pil: return None
    # ensure_rgb_or_rgba est appelé avant
    r, g, b, a = image_pil.split() # Splitte en 4 canaux (RGBA)

    channels_rgb = [r, g, b]
    random.shuffle(channels_rgb) # Mélange seulement R, G, B

    # Fusionner les canaux mélangés avec le canal alpha original
    img_glitched = Image.merge("RGBA", (*channels_rgb, a)) # Utilise * pour dépaqueter la liste mélangée
    return img_glitched


def glitch_horizontal_tear(image_pil: Image.Image, segment_height=10, max_shift=50, num_segments=None) -> Image.Image:
    """
    Glitch de déchirure horizontale. Décale des segments horizontaux de l'image.

    Args:
        image_pil: L'image PIL d'entrée (attendu en mode RGBA).
        segment_height: La hauteur de chaque segment.
        max_shift: Le décalage horizontal maximal (positif ou négatif).
        num_segments: Le nombre approximatif de segments à décaler. None ou 0 pour tous.

    Returns:
        Une nouvelle image PIL glitchée (RGBA).
    """
    if not image_pil: return None
    # ensure_rgb_or_rgba est appelé avant
    img_np = np.array(image_pil) # RGBA -> (H, W, 4)
    width, height, channels = img_np.shape

    segment_height = max(1, int(segment_height))
    max_shift = max(0, int(max_shift))

    num_total_segments = height // segment_height
    if num_total_segments == 0:
        print("Hauteur de segment trop grande pour la déchirure.")
        return image_pil.copy() # Retourne une copie si pas de changement possible

    # Déterminer quels segments décaler
    num_segments_param = int(num_segments) if num_segments is not None else -1

    if num_segments_param <= 0 or num_segments_param >= num_total_segments:
         segments_to_shift_indices = list(range(num_total_segments)) # Décaler tous les segments
    else:
         segments_to_shift_indices = random.sample(range(num_total_segments), num_segments_param)

    img_glitched_np = img_np.copy() # Copie pour modifier

    # Appliquer le décalage à chaque segment sélectionné
    for i in segments_to_shift_indices:
        y_start = i * segment_height
        y_end = min(y_start + segment_height, height)

        # Sélectionner la vue du segment et la décaler circulairement
        segment_view = img_glitched_np[y_start:y_end, :, :]
        shift = random.randint(-max_shift, max_shift)
        if shift != 0:
            # Appliquer np.roll sur l'axe horizontal (axe 1) pour tous les canaux
            shifted_segment = np.roll(segment_view, shift, axis=1)
            # Copier les données décalées dans le tableau original (via la vue)
            img_glitched_np[y_start:y_end, :, :] = shifted_segment

    return Image.fromarray(img_glitched_np, image_pil.mode) # Retourne en RGBA


def glitch_block_shuffle(image_pil: Image.Image, block_size=20, num_shuffles=100) -> Image.Image:
    """
    Glitch de mélange de blocs. Échange aléatoirement des blocs carrés de l'image.

    Args:
        image_pil: L'image PIL d'entrée (attendu en mode RGBA).
        block_size: La taille en pixels (côté du carré) des blocs.
        num_shuffles: Le nombre d'échanges de paires de blocs à effectuer.

    Returns:
        Une nouvelle image PIL glitchée (RGBA).
    """
    if not image_pil: return None
    img = image_pil.copy() # Travailler sur une copie car on utilise PIL crop/paste qui modifie l'image
    width, height = img.size

    block_size = max(1, int(block_size))
    num_shuffles = max(0, int(num_shuffles))

    # Déterminer le nombre de blocs possibles (tronqué)
    num_blocks_x = width // block_size
    num_blocks_y = height // block_size

    if num_blocks_x == 0 or num_blocks_y == 0:
        print("Taille de bloc trop grande ou image trop petite pour le mélange de blocs.")
        return img # Retourne une copie non modifiée

    # Créer une liste de coordonnées (x, y) du coin supérieur gauche de chaque bloc
    block_coords = []
    for j in range(num_blocks_y): # y
        for i in range(num_blocks_x): # x
            block_coords.append((i * block_size, j * block_size))

    if len(block_coords) < 2 or num_shuffles == 0:
         print("Pas assez de blocs pour mélanger ou 0 mélanges demandés.")
         return img


    # Effectuer les échanges de blocs
    # On choisit un nombre suffisant d'indices aléatoires pour couvrir le nombre de mélanges demandés
    num_indices_needed = min(num_shuffles * 2, len(block_coords) * 2) # Max 2*N indices, ou 2*MaxSwapsPossibles
    if num_indices_needed == 0: return img

    # Choisir des indices aléatoires dans la liste des coordonnées des blocs
    block_indices_to_swap = random.sample(range(len(block_coords)), num_indices_needed)

    # Effectuer les mélanges par paires d'indices sélectionnés
    for i in range(num_shuffles):
        # S'assurer qu'on a une paire d'indices disponible
        if 2*i + 1 >= len(block_indices_to_swap):
             break # Plus assez d'indices pour former une nouvelle paire

        idx1 = block_indices_to_swap[2*i]
        idx2 = block_indices_to_swap[2*i + 1]

        # Obtenir les coordonnées des coins supérieurs gauches
        (x1, y1) = block_coords[idx1]
        (x2, y2) = block_coords[idx2]

        if (x1, y1) == (x2, y2): # Si les blocs sont les mêmes, passer
            continue

        # Définir les régions (boxes) des blocs (x_start, y_start, x_end, y_end)
        box1 = (x1, y1, x1 + block_size, y1 + block_size)
        box2 = (x2, y2, x2 + block_size, y2 + block_size)

        try:
            # Cropper les deux blocs
            region1 = img.crop(box1)
            region2 = img.crop(box2)

            # Coller le bloc 2 à la place du bloc 1
            img.paste(region2, box1)
            # Coller le bloc 1 à la place du bloc 2
            img.paste(region1, box2)
        except Exception as e:
            print(f"Erreur lors du mélange de blocs ({i+1}/{num_shuffles}): {e}")
            continue # Continuer avec les autres mélanges si possible

    return img # Retourne l'image modifiée


def glitch_byte_segment_shuffle(image_pil: Image.Image, segment_size=1000, num_shuffles=500) -> Image.Image:
    """
    Glitch de mélange de segments de bytes. Divise les données de pixels en segments de bytes
    et les mélange aléatoirement.

    Args:
        image_pil: L'image PIL d'entrée (attendu en mode RGBA).
        segment_size: La taille en bytes de chaque segment.
        num_shuffles: Le nombre d'échanges de paires de segments à effectuer.

    Returns:
        Une nouvelle image PIL glitchée (RGBA). Retourne l'original si échec critique.
    """
    if not image_pil: return None
    img = image_pil.copy() # Copie l'image PIL
    img_bytes = bytearray(img.tobytes()) # Obtient les bytes (en RGBA)
    data_len = len(img_bytes)
    if data_len == 0: return img # Rien à faire si pas de données

    segment_size = max(1, int(segment_size))
    num_shuffles = max(0, int(num_shuffles))

    # Créer une liste de *copies* des segments de bytes
    segments = [img_bytes[i:i + segment_size] for i in range(0, data_len, segment_size)]
    num_segments = len(segments)

    if num_segments < 2 or num_shuffles == 0:
        print("Pas assez de segments de bytes pour mélanger ou 0 mélanges demandés.")
        return img # Retourne une copie non modifiée


    # Effectuer les mélanges de segments en échangeant des éléments dans la liste 'segments'
    # On choisit un nombre suffisant d'indices aléatoires pour les mélanges
    num_indices_needed = min(num_shuffles * 2, num_segments * 2)
    if num_indices_needed == 0: return img

    segment_indices_to_swap = random.sample(range(num_segments), num_indices_needed)

    for i in range(num_shuffles):
        if 2*i + 1 >= len(segment_indices_to_swap):
             break # Plus assez d'indices

        idx1 = segment_indices_to_swap[2*i]
        idx2 = segment_indices_to_swap[2*i + 1]

        if idx1 == idx2: # Ne pas échanger un segment avec lui-même
            continue

        # Échanger les segments (références dans la liste)
        segments[idx1], segments[idx2] = segments[idx2], segments[idx1]


    # Reconstruire la séquence d'octets à partir de la liste de segments mélangés
    reconstructed_bytes = bytearray()
    for seg in segments: # Les éléments dans 'segments' ont été réarrangés
        reconstructed_bytes.extend(seg)

    # La longueur devrait être la même que data_len car on a juste réarrangé les segments
    if len(reconstructed_bytes) != data_len:
        print(f"Erreur interne: La taille des bytes reconstruite ({len(reconstructed_bytes)}) diffère de l'originale ({data_len}).")
        # Cela indique un problème dans la logique de segmentation/reconstruction si ça arrive.

    # Reconvertir en image PIL
    try: return Image.frombytes(img.mode, img.size, bytes(reconstructed_bytes))
    except ValueError as e:
        print(f"Erreur: Les données corrompues après mélange de segments de bytes ne peuvent pas être reconstruites. {e}")
        messagebox.showwarning("Glitch Échoué", "Le mélange de segments de données a rendu l'image invalide.")
        return image_pil # Retourne l'originale non corrompue


def glitch_channel_data_bend(image_pil: Image.Image, num_modifications_per_channel=1000, max_byte_change=100) -> Image.Image:
     """
     Glitch de corruption par canal. Applique une modification aléatoire d'octets
     indépendamment sur chaque canal RGB.

     Args:
         image_pil: L'image PIL d'entrée (attendu en mode RGBA).
         num_modifications_per_channel: Le nombre d'octets à modifier dans CHAQUE canal RGB.
         max_byte_change: La variation maximale lors de la modification d'un octet.

     Returns:
         Une nouvelle image PIL glitchée (RGBA). Retourne l'original si échec critique.
     """
     if not image_pil: return None
     # On travaille sur les canaux R, G, B, donc on convertit en RGB pour s'assurer d'en avoir 3
     # et on gérera le canal Alpha à part.
     img = image_pil.convert("RGB")
     alpha_channel = image_pil.getchannel('A') if 'A' in image_pil.getbands() else None # Récupère le canal alpha original

     r, g, b = img.split() # Séparer les canaux RGB

     channels = [r, g, b]
     glitched_channels_rgb = []

     num_modifications_per_channel = max(0, int(num_modifications_per_channel))
     max_byte_change = max(0, min(int(max_byte_change), 255))


     for i, channel in enumerate(channels):
         channel_bytes = bytearray(channel.tobytes())
         data_len = len(channel_bytes)
         if data_len == 0:
             glitched_channels_rgb.append(channel) # Ajouter le canal non modifié s'il est vide
             continue

         num_mods = min(num_modifications_per_channel, data_len) # Limiter mods par canal

         # Sélectionner des positions aléatoires dans le canal
         if num_mods > data_len // 2:
             positions_to_modify = list(range(data_len))
             random.shuffle(positions_to_modify)
             positions_to_modify = positions_to_modify[:num_mods]
         else:
             try:
                 positions_to_modify = random.sample(range(data_len), num_mods)
             except ValueError:
                 positions_to_modify = []


         for pos in positions_to_modify:
             # Appliquer un changement aléatoire à l'octet (add_random type)
             change = random.randint(-max_byte_change, max_byte_change)
             channel_bytes[pos] = (channel_bytes[pos] + change) % 256


         try:
             # Reconstruire le canal à partir des bytes modifiés (mode='L' car c'est un canal unique de luminance)
             glitched_channel = Image.frombytes('L', img.size, bytes(channel_bytes))
             glitched_channels_rgb.append(glitched_channel)
         except ValueError:
              print(f"Erreur lors de la reconstruction du canal {['R','G','B'][i]} glitché. Utilisation du canal original.")
              glitched_channels_rgb.append(channel) # Utiliser le canal original en cas d'erreur


     # S'assurer qu'on a bien 3 canaux RGB pour la fusion
     while len(glitched_channels_rgb) < 3:
          print("Manque de canaux RGB après glitch par canal! Ajout d'un canal de secours noir.")
          width, height = img.size
          glitched_channels_rgb.append(Image.new('L', (width, height), 0)) # Ajouter un canal noir


     try:
         # Fusionner les 3 canaux RGB glitchés
         img_glitched_rgb = Image.merge("RGB", glitched_channels_rgb)

         # Ajouter le canal alpha original si l'image originale en avait un
         if alpha_channel:
              img_glitched_rgba = img_glitched_rgb.copy() # Copier l'image RGB
              img_glitched_rgba.putalpha(alpha_channel) # Ajouter le canal alpha
              return img_glitched_rgba # Retourne en RGBA
         else:
              return img_glitched_rgb # Retourne en RGB si pas de canal alpha original

     except ValueError as e:
         print(f"Erreur lors de la fusion des canaux glitchés: {e}")
         messagebox.showwarning("Glitch Échoué", "La fusion des canaux glitchés a échoué.")
         return image_pil # Retourne l'originale non corrompue


# --- Définition des Glitches et de leurs paramètres pour l'UI ---
# Chaque entrée: 'Nom affiché': {'func': fonction_glitch, 'params': [{...}]}
# Dans params: {'name': 'nom_param_python', 'label': 'Nom affiché UI', 'type': 'slider'/'option', ...}

GLITCH_DEFINITIONS = {
    "Aucun": {
        'func': None, # Pas de fonction pour "Aucun"
        'params': []
    },
    "Tri de pixels": {
        'func': glitch_pixel_sort,
        'params': [
            {'name': 'threshold', 'label': 'Seuil', 'type': 'slider', 'min': 0, 'max': 255, 'default': 100, 'resolution': 1},
            {'name': 'sort_by', 'label': 'Trier par', 'type': 'option', 'options': ['luminosity', 'red', 'green', 'blue'], 'default': 'luminosity'},
            {'name': 'direction', 'label': 'Direction', 'type': 'option', 'options': ['horizontal', 'vertical'], 'default': 'horizontal'}
        ]
    },
    "Destruction de données (bytes)": {
        'func': glitch_byte_destruction,
        'params': [
            {'name': 'num_bytes_to_modify', 'label': 'Nb octets à modifier', 'type': 'slider', 'min': 100, 'max': 500000, 'default': 10000, 'resolution': 1}, # Augmenté max
            {'name': 'destruction_type', 'label': 'Type de destruction', 'type': 'option', 'options': ['zero', 'random', 'invert', 'add_random', 'swap'], 'default': 'zero'},
            {'name': 'max_byte_change', 'label': 'Changement max (si add_random)', 'type': 'slider', 'min': 1, 'max': 255, 'default': 100, 'resolution': 1}
        ]
    },
    "Mélange de canaux RGB": {
        'func': glitch_channel_shuffle,
        'params': [] # Pas de paramètres spécifiques
    },
    "Déchirure horizontale": {
        'func': glitch_horizontal_tear,
        'params': [
             {'name': 'segment_height', 'label': 'Hauteur segment', 'type': 'slider', 'min': 1, 'max': 100, 'default': 10, 'resolution': 1}, # Augmenté max_size
             {'name': 'max_shift', 'label': 'Décalage max', 'type': 'slider', 'min': 0, 'max': 300, 'default': 50, 'resolution': 1}, # Augmenté max_shift
             {'name': 'num_segments', 'label': 'Nb segments (0=tous)', 'type': 'slider', 'min': 0, 'max': 300, 'default': 30, 'resolution': 1} # Augmenté max, renommé label
        ]
    },
    "Mélange de blocs (visuel)": {
        'func': glitch_block_shuffle,
        'params': [
            {'name': 'block_size', 'label': 'Taille du bloc', 'type': 'slider', 'min': 5, 'max': 200, 'default': 20, 'resolution': 1}, # Augmenté max
            {'name': 'num_shuffles', 'label': 'Nb mélanges', 'type': 'slider', 'min': 10, 'max': 2000, 'default': 100, 'resolution': 1} # Augmenté max
        ]
    },
    "Mélange de segments de bytes": {
        'func': glitch_byte_segment_shuffle,
        'params': [
             {'name': 'segment_size', 'label': 'Taille segment (bytes)', 'type': 'slider', 'min': 10, 'max': 50000, 'default': 1000, 'resolution': 1}, # Ajusté min/max
             {'name': 'num_shuffles', 'label': 'Nb mélanges', 'type': 'slider', 'min': 10, 'max': 5000, 'default': 500, 'resolution': 1} # Ajusté max
        ]
    },
    "Corruption par canal RGB": {
        'func': glitch_channel_data_bend,
        'params': [
            {'name': 'num_modifications_per_channel', 'label': 'Nb modif / canal', 'type': 'slider', 'min': 100, 'max': 100000, 'default': 1000, 'resolution': 1}, # Augmenté max
            {'name': 'max_byte_change', 'label': 'Changement max (octet)', 'type': 'slider', 'min': 1, 'max': 255, 'default': 100, 'resolution': 1}
        ]
    }
    # Ajoutez vos autres glitches ici
}


# --- Application GUI ---

class GlitchApp:
    def __init__(self, root):
        self.root = root
        root.title("Applicateur de Glitch Destructeur")
        root.geometry("900x750") # Taille initiale légèrement augmentée
        root.option_add('*tearOff', tk.FALSE) # Désactiver les menus "déchirables"

        self.image_originale = None # Image PIL RGBA originale chargée
        self.image_actuelle = None # Image PIL RGBA actuellement modifiée
        self.photo_image_tk = None # Image formatée pour Tkinter display

        # --- Widgets ---
        # Utilisation de Frame et Grid pour organiser la mise en page
        self.frame_commandes = ttk.Frame(root, padding="10") # Ajout de padding
        self.frame_commandes.pack(pady=10, padx=10, fill='x') # Prend la largeur

        # Frame pour les boutons de contrôle principaux
        frame_controles_principaux = ttk.Frame(self.frame_commandes)
        frame_controles_principaux.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10)) # S'étend sur 2 colonnes

        ttk.Button(frame_controles_principaux, text="Charger photo", command=self.charger_image).pack(side='left', padx=5)

        ttk.Label(frame_controles_principaux, text="Glitch:").pack(side='left', padx=5)
        self.variable_glitch = tk.StringVar(root)
        self.variable_glitch.set(list(GLITCH_DEFINITIONS.keys())[0])
        self.menu_glitch = ttk.OptionMenu(frame_controles_principaux, self.variable_glitch, self.variable_glitch.get(), *GLITCH_DEFINITIONS.keys(), command=self.update_parameter_ui)
        self.menu_glitch.pack(side='left', padx=5, fill='x', expand=True) # Prend la place restante

        ttk.Button(frame_controles_principaux, text="Appliquer", command=self.appliquer_glitch).pack(side='left', padx=5)
        ttk.Button(frame_controles_principaux, text="Reset", command=self.reset_image).pack(side='left', padx=5)
        ttk.Button(frame_controles_principaux, text="Enregistrer", command=self.enregistrer_image).pack(side='left', padx=5)


        # Frame pour les paramètres dynamiques du glitch sélectionné
        # Utilise grid pour se placer sous les contrôles principaux
        self.frame_parametres = ttk.LabelFrame(self.frame_commandes, text="Paramètres du Glitch", padding="10") # Ajout de padding
        self.frame_parametres.grid(row=1, column=0, columnspan=2, sticky="ew") # S'étend sur 2 colonnes, prend la largeur


        # Label pour afficher l'image
        # Utilise pack, il sera sous le frame_commandes
        # Le relief aide à visualiser la zone de l'image
        self.label_image = ttk.Label(root, relief="groove")
        self.label_image.pack(pady=10, padx=10) # Ne pas utiliser fill/expand ici si on veut que l'image soit centrée et non étirée

        # Barre de statut
        self.status_bar = ttk.Label(root, text="Charger une image pour commencer.", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X) # Prend toute la largeur en bas


        # Initialiser l'interface des paramètres pour le glitch par défaut
        self.update_parameter_ui()
        # Mettre à jour l'état des boutons au démarrage (aucun image chargée)
        self.update_button_states()


    # --- Gestion des paramètres dynamiques ---
    def update_parameter_ui(self, *args):
        """ Met à jour les widgets de paramètres en fonction du glitch sélectionné. """
        # Effacer les widgets précédents
        for widget in self.frame_parametres.winfo_children():
            widget.destroy()
        self.parameter_widgets = {} # Réinitialise le dictionnaire de widgets

        selected_glitch_name = self.variable_glitch.get()
        glitch_info = GLITCH_DEFINITIONS.get(selected_glitch_name)

        if not glitch_info or not glitch_info['params']:
            ttk.Label(self.frame_parametres, text="Aucun paramètre pour ce glitch.").pack(padx=5, pady=5)
            return

        # Utilise grid pour organiser les paramètres en lignes
        row_idx = 0
        for param_info in glitch_info['params']:
            frame_param = ttk.Frame(self.frame_parametres)
            frame_param.grid(row=row_idx, column=0, sticky="ew", pady=2) # Chaque paramètre sur une nouvelle ligne

            # Utiliser la clé 'label' pour le texte affiché, mais 'name' pour la clé stockée
            label_text = param_info.get('label', param_info['name']) # Utilise 'label' si défini, sinon 'name'
            ttk.Label(frame_param, text=label_text + ":").pack(side='left', padx=(0, 5)) # Padding à droite

            if param_info['type'] == 'slider':
                # Utilise Scale (Tkinter standard)
                # Déterminer le type de variable et la résolution
                resolution = param_info.get('resolution', 1)
                if isinstance(param_info.get('default', 0), int) and resolution == 1:
                     scale_var = tk.IntVar(value=param_info['default'])
                else:
                     scale_var = tk.DoubleVar(value=param_info['default'])
                     # Si la résolution n'est pas 1, s'assurer que c'est un float
                     if resolution == 1: resolution = 1.0


                slider = tk.Scale(frame_param, from_=param_info['min'], to=param_info['max'],
                                  orient=HORIZONTAL, variable=scale_var,
                                  length=250, # Longueur fixe pour alignement
                                  resolution=resolution,
                                  command=self.update_status_param_preview) # Optionnel: afficher valeur en direct
                slider.pack(side='left', fill='x', expand=True) # Prend la place restante dans la frame_param
                # Stocke la variable Tkinter en utilisant le NOM DU PARAMÈTRE PYTHON ('name')
                self.parameter_widgets[param_info['name']] = scale_var

            elif param_info['type'] == 'option':
                option_var = tk.StringVar(value=param_info['default'])
                # Le * dépaquette la liste d'options pour les passer comme arguments individuels
                option_menu = ttk.OptionMenu(frame_param, option_var, option_var.get(), *param_info['options'], command=self.update_status_param_preview)
                option_menu.pack(side='left', padx=5, fill='x', expand=True)
                 # Stocke la variable Tkinter en utilisant le NOM DU PARAMÈTRE PYTHON ('name')
                self.parameter_widgets[param_info['name']] = option_var

            row_idx += 1 # Passe à la ligne suivante pour le prochain paramètre

        # Configurer la colonne 0 de frame_parametres pour s'étendre
        self.frame_parametres.grid_columnconfigure(0, weight=1)


    def get_glitch_parameters(self):
        """
        Récupère les valeurs actuelles des paramètres depuis les widgets de l'UI.

        Returns:
            Un dictionnaire {nom_param_python: valeur} des paramètres.
        """
        params = {}
        selected_glitch_name = self.variable_glitch.get()
        glitch_info = GLITCH_DEFINITIONS.get(selected_glitch_name)

        if not glitch_info: return params

        for param_info in glitch_info['params']:
            # Utilise le NOM DU PARAMÈTRE PYTHON ('name') pour récupérer la variable stockée
            param_name = param_info['name']
            widget_var = self.parameter_widgets.get(param_name)
            if widget_var:
                value = widget_var.get()
                # Convertir en int ou float si nécessaire, basé sur la résolution du slider
                if param_info['type'] == 'slider':
                     resolution = param_info.get('resolution', 1)
                     if resolution == 1 and isinstance(value, float) and value.is_integer():
                         params[param_name] = int(value)
                     else:
                         params[param_name] = value
                else: # string for option menus
                    params[param_name] = value

        return params

    def update_status(self, message):
        """ Met à jour le texte de la barre de statut. """
        self.status_bar.config(text=message)

    def update_status_param_preview(self, value=None):
        """ Met à jour le statut avec un aperçu du paramètre en cours d'ajustement. """
        selected_glitch_name = self.variable_glitch.get()
        # Tentative d'afficher une info simple sur le paramètre ajusté
        # C'est un peu tricky car la commande du slider ne donne que la valeur, pas le nom du param
        # On pourrait passer plus d'info via une lambda dans update_parameter_ui
        # Pour l'instant, juste afficher le nom du glitch et la valeur si disponible
        if value is not None:
             self.update_status(f"{selected_glitch_name} - Ajustement: {value}")
        else:
             self.update_status(f"Paramètres pour: {selected_glitch_name}")


    def update_button_states(self):
        """ Active ou désactive les boutons selon si une image est chargée. """
        is_image_loaded = self.image_actuelle is not None
        # Le bouton "Charger" est toujours actif
        # Les autres boutons (Appliquer, Reset, Enregistrer) ne sont actifs que si une image est chargée
        for widget in self.frame_commandes.winfo_children():
            # On cherche spécifiquement les boutons "Appliquer", "Reset", "Enregistrer"
            if isinstance(widget, ttk.Frame): # Les boutons sont dans frame_controles_principaux
                for btn in widget.winfo_children():
                     if isinstance(btn, ttk.Button):
                         btn_text = btn.cget('text')
                         if btn_text in ["Appliquer", "Reset", "Enregistrer"]:
                              btn.config(state=tk.NORMAL if is_image_loaded else tk.DISABLED)
                break # On a trouvé la frame, on peut sortir


    # --- Gestion de l'image et des actions ---

    def charger_image(self):
        file_path = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Sélectionner un fichier image",
            filetypes=IMAGE_FILETYPES
        )
        if file_path:
            try:
                self.update_status(f"Chargement de {os.path.basename(file_path)}...")
                with Image.open(file_path) as img_pil:
                    # Convertir immédiatement en RGBA pour une base de travail cohérente
                    self.image_originale = ensure_rgb_or_rgba(img_pil.copy())
                    self.image_actuelle = self.image_originale.copy() # Toujours commencer par l'original RGBA
                self.afficher_image(self.image_actuelle)
                self.variable_glitch.set(list(GLITCH_DEFINITIONS.keys())[0]) # Reset glitch selection
                self.update_parameter_ui() # Mettre à jour les paramètres
                self.update_button_states() # Activer les boutons
                self.update_status(f"'{os.path.basename(file_path)}' chargé. Prêt à glitcher.")
            except Exception as e:
                self.update_status(f"Erreur de chargement.")
                messagebox.showerror("Erreur de chargement", f"Impossible de charger l'image: {e}")
                self.reset_image() # Nettoyer l'état


    def afficher_image(self, image_pil: Image.Image):
        """ Redimensionne et affiche l'image PIL dans le Label Tkinter. """
        if not image_pil:
            self.label_image.config(image="")
            self.photo_image_tk = None
            return

        # Redimensionner l'image pour l'affichage
        # Utiliser thumbnail pour garder le ratio et ne pas dépasser MAX_DISPLAY_SIZE
        img_pil_display = image_pil.copy() # Copie avant redimensionnement
        img_pil_display.thumbnail(MAX_DISPLAY_SIZE, Resampling) # Utilise la méthode de resampling définie

        try:
             # Convertir en un mode compatible PhotoImage si ce n'est pas déjà le cas
             # ensure_rgb_or_rgba garantit que self.image_actuelle est RGBA, mais thumbnail pourrait changer mode si pas de alpha?
             # Mieux vaut reconvertir explicitement pour PhotoImage si on n'est pas certain.
             # PhotoImage supporte 'RGB', 'RGBA', 'L', 'P'. RGBA est le plus universel pour gérer transparence.
             if img_pil_display.mode not in ['RGB', 'RGBA']:
                  img_pil_display = img_pil_display.convert('RGBA')

             self.photo_image_tk = ImageTk.PhotoImage(img_pil_display)
             self.label_image.config(image=self.photo_image_tk)
             self.label_image.image = self.photo_image_tk # Prévenir la suppression par le GC
        except Exception as e:
             print(f"Erreur lors de l'affichage de l'image : {e}")
             import traceback
             traceback.print_exc()
             self.label_image.config(image="")
             self.photo_image_tk = None
             messagebox.showwarning("Erreur Affichage", "Impossible d'afficher l'image modifiée.")


    def appliquer_glitch(self):
        """ Applique le glitch sélectionné avec les paramètres actuels. """
        if not self.image_actuelle:
            messagebox.showwarning("Attention", "Aucune image chargée.")
            self.update_status("Impossible d'appliquer : aucune image chargée.")
            return

        selected_glitch_name = self.variable_glitch.get()
        glitch_info = GLITCH_DEFINITIONS.get(selected_glitch_name)

        if not glitch_info or glitch_info['func'] is None:
             # Option "Aucun" sélectionnée
             if self.image_originale:
                 # Revenir à l'image originale RGBA pour une base propre
                 self.image_actuelle = self.image_originale.copy()
                 self.afficher_image(self.image_actuelle)
                 self.update_status("Image réinitialisée à l'original.")
             return # Ne rien faire si "Aucun" et pas d'original

        params = self.get_glitch_parameters()
        self.update_status(f"Application de '{selected_glitch_name}'...")
        print(f"Appliquer glitch: {selected_glitch_name} avec paramètres: {params}") # Debug

        try:
            # Appliquer le glitch sur une copie de l'image actuelle
            # S'assurer que la fonction de glitch reçoit une image RGBA
            img_glitched = glitch_info['func'](self.image_actuelle.copy(), **params)

            if img_glitched and isinstance(img_glitched, Image.Image):
                # La fonction de glitch devrait retourner une image en mode RGB ou RGBA
                self.image_actuelle = img_glitched # Met à jour l'image actuelle
                self.afficher_image(self.image_actuelle)
                self.update_status(f"'{selected_glitch_name}' appliqué avec succès.")
            else:
                 self.update_status(f"'{selected_glitch_name}' a échoué ou n'a pas retourné d'image valide.")
                 messagebox.showwarning("Glitch Échoué", f"Le glitch '{selected_glitch_name}' n'a pas retourné une image valide.")
                 # L'image_actuelle reste la même qu'avant l'application

        except Exception as e:
            self.update_status(f"Erreur lors de l'application de '{selected_glitch_name}'.")
            messagebox.showerror("Erreur Glitch", f"Une erreur s'est produite lors de l'application du glitch: {e}")
            import traceback
            traceback.print_exc()


    def reset_image(self):
        """ Revenir à l'image originale chargée. """
        if self.image_originale:
            self.image_actuelle = self.image_originale.copy()
            self.afficher_image(self.image_actuelle)
            self.variable_glitch.set(list(GLITCH_DEFINITIONS.keys())[0]) # Reset glitch selection
            self.update_parameter_ui() # Mettre à jour les paramètres
            self.update_status("Image réinitialisée à l'original.")
        else:
            # Si pas d'original (n'a jamais chargé ou reset après un échec de chargement)
            self.image_actuelle = None
            self.image_originale = None
            self.afficher_image(None)
            self.update_button_states() # Désactiver les boutons
            self.update_status("Aucune image chargée.")


    def enregistrer_image(self):
        """ Enregistre l'image actuellement affichée dans un fichier. """
        if not self.image_actuelle:
            messagebox.showwarning("Attention", "Aucune image à enregistrer.")
            self.update_status("Impossible d'enregistrer : aucune image chargée.")
            return

        file_path = filedialog.asksaveasfilename(
            initialdir=os.getcwd(),
            title="Enregistrer l'image glitchée",
            defaultextension=DEFAULT_SAVE_EXTENSION,
            filetypes=IMAGE_FILETYPES
        )
        if file_path:
            try:
                self.update_status(f"Enregistrement de '{os.path.basename(file_path)}'...")
                img_to_save = self.image_actuelle.copy() # Travailler sur une copie pour la sauvegarde

                # Pillow gère la conversion de mode pour la sauvegarde en fonction de l'extension
                # S'assurer que l'image est dans un mode géré par Pillow pour la sauvegarde
                # RGBA est généralement sûr et gère la transparence si présente.
                # Convertir en RGBA si le mode actuel n'est pas un mode standard de sauvegarde
                if img_to_save.mode not in ['RGB', 'RGBA', 'L', 'P', 'CMYK', 'YCbCr', 'HSV', 'I', 'F']:
                     img_to_save = img_to_save.convert('RGBA') # Convertir si besoin


                img_to_save.save(file_path)
                self.update_status(f"Image enregistrée sous '{os.path.basename(file_path)}'.")
                messagebox.showinfo("Succès", "Image enregistrée avec succès!")
            except Exception as e:
                self.update_status(f"Erreur lors de l'enregistrement.")
                messagebox.showerror("Erreur d'enregistrement", f"Impossible d'enregistrer l'image: {e}")
                import traceback
                traceback.print_exc()


# --- Lancement de l'application ---
if __name__ == "__main__":
    root = tk.Tk()
    app = GlitchApp(root)
    root.mainloop()