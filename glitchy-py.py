import tkinter as tk
from tkinter import filedialog, messagebox, Scale, HORIZONTAL, ttk
from PIL import Image, ImageTk, ImageFilter
import numpy as np
import random
import os
import io
import sys
import math
import time

# --- Constantes de configuration ---
MAX_DISPLAY_SIZE = (800, 600)
IMAGE_FILETYPES = [
    ("Fichiers image", "*.png *.jpg *.jpeg *.gif *.bmp"),
    ("Tous les fichiers", "*.*")
]
DEFAULT_SAVE_EXTENSION = ".png"

try:
    Resampling = Image.Resampling.LANCZOS
except AttributeError:
    Resampling = Image.LANCZOS

# Limiter l'historique pour ne pas consommer trop de mémoire
MAX_HISTORY_SIZE = 20

# --- Fonctions utilitaires ---

def ensure_rgb_or_rgba(image_pil: Image.Image) -> Image.Image:
    """
    Assure que l'image est en mode RGBA.
    """
    if image_pil.mode != 'RGBA':
        return image_pil.convert('RGBA')
    return image_pil

# --- Fonctions d'algorithmes de Glitch ---
# (Code des fonctions de glitch omis pour la concision, il est le même que précédemment)

def glitch_pixel_sort(image_pil: Image.Image, threshold=100, sort_by='luminosity', direction='horizontal') -> Image.Image:
    """Glitch de tri de pixels."""
    if not image_pil: return None
    img_np = np.array(image_pil)
    height, width, channels = img_np.shape
    threshold = max(0, min(int(threshold), 255))
    if sort_by == 'luminosity': base_sort = np.sum(img_np[:,:,:3].astype(np.float32) * [0.299, 0.587, 0.114], axis=2)
    elif sort_by == 'red': base_sort = img_np[:,:,0]
    elif sort_by == 'green': base_sort = img_np[:,:,1]
    elif sort_by == 'blue': base_sort = img_np[:,:,2]
    else: base_sort = np.sum(img_np[:,:,:3].astype(np.float32) * [0.299, 0.587, 0.114], axis=2)
    img_glitched_np = img_np.copy()
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
    return Image.fromarray(img_glitched_np, image_pil.mode)

def glitch_byte_destruction(image_pil: Image.Image, num_bytes_to_modify=10000, destruction_type='zero', max_byte_change=100) -> Image.Image:
    """Glitch de destruction d'octets."""
    if not image_pil: return None
    img = image_pil.copy()
    img_bytes = bytearray(img.tobytes())
    data_len = len(img_bytes)
    if data_len == 0: return img
    num_bytes_to_modify = max(0, min(int(num_bytes_to_modify), data_len))
    max_byte_change = max(0, min(int(max_byte_change), 255))
    if num_bytes_to_modify > data_len // 2:
         all_positions = list(range(data_len))
         random.shuffle(all_positions)
         positions_to_modify = all_positions[:num_bytes_to_modify]
    else:
         try: positions_to_modify = random.sample(range(data_len), num_bytes_to_modify)
         except ValueError: positions_to_modify = []
    if destruction_type == 'zero':
        for pos in positions_to_modify: img_bytes[pos] = 0
    elif destruction_type == 'random':
         for pos in positions_to_modify: img_bytes[pos] = random.randint(0, 255)
    elif destruction_type == 'invert':
         for pos in positions_to_modify: img_bytes[pos] = 255 - img_bytes[pos]
    elif destruction_type == 'add_random':
         for pos in positions_to_modify:
            change = random.randint(-max_byte_change, max_byte_change)
            img_bytes[pos] = (img_bytes[pos] + change) % 256
    elif destruction_type == 'swap':
         num_swaps = len(positions_to_modify) // 2
         if num_swaps > 0:
            swap_indices = random.sample(range(len(positions_to_modify)), num_swaps * 2)
            for i in range(num_swaps):
                pos1 = positions_to_modify[swap_indices[2*i]]
                pos2 = positions_to_modify[swap_indices[2*i + 1]]
                img_bytes[pos1], img_bytes[pos2] = img_bytes[pos2], img_bytes[pos1]
    try: return Image.frombytes(img.mode, img.size, bytes(img_bytes))
    except ValueError:
        print("Erreur: Les données corrompues ne peuvent pas être reconstruites par Image.frombytes.")
        messagebox.showwarning("Glitch Échoué", "La corruption de données a rendu l'image invalide.")
        return image_pil

def glitch_channel_shuffle(image_pil: Image.Image) -> Image.Image:
    """Glitch de mélange de canaux."""
    if not image_pil: return None
    r, g, b, a = image_pil.split()
    channels_rgb = [r, g, b]
    random.shuffle(channels_rgb)
    img_glitched = Image.merge("RGBA", (*channels_rgb, a))
    return img_glitched

def glitch_horizontal_tear(image_pil: Image.Image, segment_height=10, max_shift=50, num_segments=None) -> Image.Image:
    """Glitch de déchirure horizontale."""
    if not image_pil: return None
    img_np = np.array(image_pil)
    width, height, channels = img_np.shape
    segment_height = max(1, int(segment_height))
    max_shift = max(0, int(max_shift))
    num_total_segments = height // segment_height
    if num_total_segments == 0:
        print("Hauteur de segment trop grande pour la déchirure.")
        return image_pil.copy()
    num_segments_param = int(num_segments) if num_segments is not None else -1
    if num_segments_param <= 0 or num_segments_param >= num_total_segments:
         segments_to_shift_indices = list(range(num_total_segments))
    else:
         segments_to_shift_indices = random.sample(range(num_total_segments), num_segments_param)
    img_glitched_np = img_np.copy()
    for i in segments_to_shift_indices:
        y_start = i * segment_height
        y_end = min(y_start + segment_height, height)
        if y_start >= height: break
        shift = random.randint(-max_shift, max_shift)
        if shift != 0:
            segment_view = img_glitched_np[y_start:y_end, :, :]
            shifted_segment = np.roll(segment_view, shift, axis=1)
            img_glitched_np[y_start:y_end, :, :] = shifted_segment
    return Image.fromarray(img_glitched_np, image_pil.mode)

def glitch_block_shuffle(image_pil: Image.Image, block_size=20, num_shuffles=100) -> Image.Image:
    """Glitch de mélange de blocs."""
    if not image_pil: return None
    img = image_pil.copy()
    width, height = img.size
    block_size = max(1, int(block_size))
    num_shuffles = max(0, int(num_shuffles))
    num_blocks_x = width // block_size
    num_blocks_y = height // block_size
    if num_blocks_x == 0 or num_blocks_y == 0:
        print("Taille de bloc trop grande ou image trop petite pour le mélange de blocs.")
        return img
    block_coords = []
    for j in range(num_blocks_y):
        for i in range(num_blocks_x): block_coords.append((i * block_size, j * block_size))
    if len(block_coords) < 2 or num_shuffles == 0:
         print("Pas assez de blocs pour mélanger ou 0 mélanges demandés.")
         return img
    num_indices_needed = min(num_shuffles * 2, len(block_coords) * 2)
    if num_indices_needed == 0: return img
    block_indices_to_swap = random.sample(range(len(block_coords)), num_indices_needed)
    for i in range(num_shuffles):
        if 2*i + 1 >= len(block_indices_to_swap): break
        idx1 = block_indices_to_swap[2*i]
        idx2 = block_indices_to_swap[2*i + 1]
        if (block_coords[idx1] == block_coords[idx2]): continue
        (x1, y1) = block_coords[idx1]
        (x2, y2) = block_coords[idx2]
        box1 = (x1, y1, x1 + block_size, y1 + block_size)
        box2 = (x2, y2, x2 + block_size, y2 + block_size)
        try:
            region1 = img.crop(box1)
            region2 = img.crop(box2)
            img.paste(region2, box1)
            img.paste(region1, box2)
        except Exception as e:
            print(f"Erreur lors du mélange de blocs ({i+1}/{num_shuffles}): {e}")
            continue
    return img

def glitch_byte_segment_shuffle(image_pil: Image.Image, segment_size=1000, num_shuffles=500) -> Image.Image:
    """Glitch de mélange de segments de bytes."""
    if not image_pil: return None
    img = image_pil.copy()
    img_bytes = bytearray(img.tobytes())
    data_len = len(img_bytes)
    if data_len == 0: return img
    segment_size = max(1, int(segment_size))
    num_shuffles = max(0, int(num_shuffles))
    segments = [img_bytes[i:i + segment_size] for i in range(0, data_len, segment_size)]
    num_segments = len(segments)
    if num_segments < 2 or num_shuffles == 0:
        print("Pas assez de segments de bytes pour mélanger ou 0 mélanges demandés.")
        return img
    shuffled_segments_list = list(segments)
    num_actual_shuffles = min(num_shuffles, num_segments * (num_segments - 1) // 2)
    if num_actual_shuffles <= 0: return img
    segment_indices_to_swap = random.sample(range(num_segments), min(num_actual_shuffles * 2, num_segments))
    for i in range(num_shuffles):
        if 2*i + 1 >= len(segment_indices_to_swap): break
        idx1 = segment_indices_to_swap[2*i]
        idx2 = segment_indices_to_swap[2*i + 1]
        if idx1 == idx2: continue
        segments[idx1], segments[idx2] = segments[idx2], segments[idx1]
    reconstructed_bytes = bytearray()
    for seg in segments: reconstructed_bytes.extend(seg)
    if len(reconstructed_bytes) != data_len:
        print(f"Erreur interne: La taille des bytes reconstruite ({len(reconstructed_bytes)}) diffère de l'originale ({data_len}).")
    try: return Image.frombytes(img.mode, img.size, bytes(reconstructed_bytes))
    except ValueError as e:
        print(f"Erreur: Les données corrompues après mélange de segments de bytes ne peuvent pas être reconstruites. {e}")
        messagebox.showwarning("Glitch Échoué", "Le mélange de segments de données a rendu l'image invalide.")
        return image_pil

def glitch_channel_data_bend(image_pil: Image.Image, num_modifications_per_channel=1000, max_byte_change=100) -> Image.Image:
     """Glitch de corruption par canal."""
     if not image_pil: return None
     img = image_pil.convert("RGB")
     alpha_channel = image_pil.getchannel('A') if 'A' in image_pil.getbands() else None
     r, g, b = img.split()
     channels = [r, g, b]
     glitched_channels_rgb = []
     num_modifications_per_channel = max(0, int(num_modifications_per_channel))
     max_byte_change = max(0, min(int(max_byte_change), 255))
     for i, channel in enumerate(channels):
         channel_bytes = bytearray(channel.tobytes())
         data_len = len(channel_bytes)
         if data_len == 0:
             glitched_channels_rgb.append(channel)
             continue
         num_mods = min(num_modifications_per_channel, data_len)
         if num_mods > data_len // 2:
             positions_to_modify = list(range(data_len))
             random.shuffle(positions_to_modify)
             positions_to_modify = positions_to_modify[:num_mods]
         else:
             try: positions_to_modify = random.sample(range(data_len), num_mods)
             except ValueError: positions_to_modify = []
         for pos in positions_to_modify:
             change = random.randint(-max_byte_change, max_byte_change)
             channel_bytes[pos] = (channel_bytes[pos] + change) % 256
         try:
             glitched_channel = Image.frombytes('L', img.size, bytes(channel_bytes))
             glitched_channels_rgb.append(glitched_channel)
         except ValueError:
              print(f"Erreur lors de la reconstruction du canal {['R','G','B'][i]} glitché. Utilisation du canal original.")
              glitched_channels_rgb.append(channel)
     while len(glitched_channels_rgb) < 3:
          print("Manque de canaux RGB après glitch par canal! Ajout d'un canal de secours noir.")
          width, height = img.size
          glitched_channels_rgb.append(Image.new('L', (width, height), 0))
     try:
         img_glitched_rgb = Image.merge("RGB", glitched_channels_rgb)
         if alpha_channel:
              img_glitched_rgba = img_glitched_rgb.copy()
              img_glitched_rgba.putalpha(alpha_channel)
              return img_glitched_rgba
         else:
              return img_glitched_rgb
     except ValueError as e:
         print(f"Erreur lors de la fusion des canaux glitchés: {e}")
         messagebox.showwarning("Glitch Échoué", "La fusion des canaux glitchés a échoué.")
         return image_pil

def glitch_hue_cycle(image_pil: Image.Image, shift_amount=50) -> Image.Image:
    """Glitch de cycle de teinte."""
    if not image_pil: return None
    img = image_pil.copy()
    img_hsv = img.convert("HSV")
    img_hsv_np = np.array(img_hsv)
    shift_amount = int(shift_amount)
    if shift_amount != 0:
         img_hsv_np[:,:,0] = np.roll(img_hsv_np[:,:,0], shift_amount)
    img_hsv_glitched_pil = Image.fromarray(img_hsv_np, "HSV")
    img_glitched = img_hsv_glitched_pil.convert("RGBA")
    return img_glitched

def glitch_mirror_tile(image_pil: Image.Image, axis='horizontal', num_repeats=2) -> Image.Image:
    """Glitch de miroir/carrelage."""
    if not image_pil: return None
    img = image_pil.copy()
    width, height = img.size
    num_repeats = max(1, int(num_repeats))
    images_to_tile = [img]
    if axis == 'horizontal': mirror_op = Image.Transpose.FLIP_LEFT_RIGHT
    elif axis == 'vertical': mirror_op = Image.Transpose.FLIP_TOP_BOTTOM
    else: mirror_op = Image.Transpose.FLIP_LEFT_RIGHT
    current_img = img
    for _ in range(num_repeats - 1):
        try:
            current_img_for_transpose = current_img.convert("RGBA")
            mirrored_img = current_img_for_transpose.transpose(mirror_op)
            images_to_tile.append(mirrored_img)
            current_img = mirrored_img
        except Exception as e:
            print(f"Erreur lors de l l'opération miroir: {e}. Arrêt de la répétition.")
            break
    if not images_to_tile: return image_pil.copy()
    if any(img_to_tile.mode != images_to_tile[0].mode for img_to_tile in images_to_tile):
         print("Attention: Modes d'image inconsistants pour le carrelage. Conversion de toutes en RGBA.")
         images_to_tile = [img_to_tile.convert("RGBA") for img_to_tile in images_to_tile]
    if axis == 'horizontal':
        total_width = sum(img_to_tile.width for img_to_tile in images_to_tile)
        max_height = max(img_to_tile.height for img_to_tile in images_to_tile)
        tiled_img = Image.new(images_to_tile[0].mode, (total_width, max_height))
        x_offset = 0
        for img_to_paste in images_to_tile:
            tiled_img.paste(img_to_paste, (x_offset, 0))
            x_offset += img_to_paste.width
    else:
        total_height = sum(img_to_tile.height for img_to_tile in images_to_tile)
        max_width = max(img_to_tile.width for img_to_tile in images_to_tile)
        tiled_img = Image.new(images_to_tile[0].mode, (max_width, total_height))
        y_offset = 0
        for img_to_paste in images_to_tile:
            tiled_img.paste(img_to_paste, (0, y_offset))
            y_offset += img_to_paste.height
    return tiled_img

def glitch_coordinate_warp(image_pil: Image.Image, warp_intensity=50.0, warp_type='sin_x', frequency=1.0) -> Image.Image:
    """Glitch de déformation de coordonnées."""
    if not image_pil: return None
    img = image_pil.copy()
    width, height = img.size
    img_np = np.array(img)
    img_out_np = np.zeros_like(img_np)
    warp_intensity = float(warp_intensity)
    frequency = float(frequency)
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    if warp_type == 'sin_x':
        new_x = x_coords
        new_y = y_coords + warp_intensity * np.sin(x_coords / width * math.pi * 2 * frequency)
    elif warp_type == 'sin_y':
        new_x = x_coords + warp_intensity * np.sin(y_coords / height * math.pi * 2 * frequency)
        new_y = y_coords
    elif warp_type == 'diag_wave':
        new_x = x_coords + warp_intensity * np.sin((x_coords + y_coords) / (width + height) * math.pi * 2 * frequency)
        new_y = y_coords + warp_intensity * np.cos((x_coords + y_coords) / (width + height) * math.pi * 2 * frequency)
    elif warp_type == 'radial':
         center_x, center_y = width / 2, height / 2
         dist = np.hypot(x_coords - center_x, y_coords - center_y)
         angle = np.arctan2(y_coords - center_y, x_coords - center_x)
         shifted_angle = angle + warp_intensity * 0.001 * np.sin(angle * frequency)
         dist = np.maximum(dist, 1e-6)
         new_x = center_x + dist * np.cos(shifted_angle)
         new_y = center_y + dist * np.sin(shifted_angle)
    else:
         new_x = x_coords
         new_y = y_coords + warp_intensity * np.sin(x_coords / width * math.pi * 2 * frequency)
    new_x = np.clip(np.round(new_x).astype(int), 0, width - 1)
    new_y = np.clip(np.round(new_y).astype(int), 0, height - 1)
    img_out_np[:, :] = img_np[new_y, new_x, :]
    return Image.fromarray(img_out_np, img.mode)

def glitch_palette_manipulation(image_pil: Image.Image, num_colors=256, manipulation_type='shuffle', num_modifications=1000) -> Image.Image:
    """Glitch de manipulation de palette."""
    if not image_pil: return None
    img = image_pil.copy()
    try:
        num_colors = max(2, min(int(num_colors), 256))
        img_paletted = img.convert("P", palette=Image.Palette.ADAPTIVE, colors=num_colors)
        palette = bytearray(img_paletted.getpalette() or [0] * 768)
        palette_len = len(palette)
        num_modifications = max(0, min(int(num_modifications), palette_len))
        if manipulation_type == 'shuffle':
             num_entries_to_shuffle = num_modifications // 3
             if num_entries_to_shuffle > 0:
                entry_indices = list(range(0, palette_len, 3))
                if len(entry_indices) > 1:
                     num_swaps = min(num_entries_to_shuffle, len(entry_indices) * (len(entry_indices) - 1) // 2)
                     swap_entry_indices = random.sample(entry_indices, min(num_swaps * 2, len(entry_indices)))
                     for i in range(num_swaps):
                         if 2*i + 1 >= len(swap_entry_indices): break
                         idx1_start = swap_entry_indices[2*i]
                         idx2_start = swap_entry_indices[2*i + 1]
                         palette[idx1_start:idx1_start+3], palette[idx2_start:idx2_start+3] = palette[idx2_start:idx2_start+3], palette[idx1_start:idx1_start+3]
        elif manipulation_type == 'random_bytes':
             positions_to_modify = random.sample(range(palette_len), num_modifications)
             for pos in positions_to_modify: palette[pos] = random.randint(0, 255)
        elif manipulation_type == 'invert_bytes':
             positions_to_modify = random.sample(range(palette_len), num_modifications)
             for pos in positions_to_modify: palette[pos] = 255 - palette[pos]
        img_paletted.putpalette(bytes(palette))
        img_glitched = img_paletted.convert("RGBA")
        return img_glitched
    except Exception as e:
        print(f"Erreur lors de la manipulation de palette: {e}")
        messagebox.showwarning("Glitch Échoué", "La manipulation de palette a échoué.")
        return image_pil

def glitch_scanlines(image_pil: Image.Image, intensity=50, line_height=2, offset=0, scanline_color_type='darken') -> Image.Image:
    """Glitch de lignes de balayage (Scanlines)."""
    if not image_pil: return None
    img = image_pil.copy()
    img_np = np.array(img)
    height, width, channels = img_np.shape
    intensity = max(0, min(int(intensity), 255))
    line_height = max(1, int(line_height))
    offset = int(offset) % line_height if line_height > 0 else 0
    mask = np.zeros(height, dtype=bool)
    if line_height > 0: mask[(offset % line_height) :: line_height] = True
    img_glitched_np = img_np.copy()
    if scanline_color_type == 'darken':
         img_glitched_np[mask, :, :3] = np.clip(img_glitched_np[mask, :, :3] - intensity, 0, 255)
    elif scanline_color_type == 'zero':
         img_glitched_np[mask, :, :3] = 0
    elif scanline_color_type == 'color':
         base_color = np.array([0, 50, 0], dtype=np.float32)
         acid_color = np.array([0, 255, 0], dtype=np.float32)
         scan_color = (base_color + (acid_color - base_color) * (intensity / 255.0)).astype(np.uint8)
         img_glitched_np[mask, :, :3] = scan_color
    return Image.fromarray(img_glitched_np, img.mode)

def glitch_iterative_simple(image_pil: Image.Image, num_iterations=10, byte_shift_amount=5) -> Image.Image:
     """Applique un petit glitch simple (décalage de bytes) N fois."""
     if not image_pil: return None
     img = image_pil.copy()
     num_iterations = max(0, int(num_iterations))
     byte_shift_amount = int(byte_shift_amount)
     def _simple_shift_step(current_img_bytes: bytearray, shift: int) -> bytearray:
         data_len = len(current_img_bytes)
         if data_len == 0 or shift == 0: return current_img_bytes
         abs_shift = abs(shift) % data_len
         if abs_shift == 0: return current_img_bytes
         if shift > 0: return current_img_bytes[-abs_shift:] + current_img_bytes[:-abs_shift]
         else: return current_img_bytes[abs_shift:] + current_img_bytes[:abs_shift]
     current_img_bytes = bytearray(img.tobytes())
     img_mode = img.mode
     img_size = img.size
     for i in range(num_iterations):
         next_img_bytes = _simple_shift_step(current_img_bytes, byte_shift_amount)
         current_img_bytes = next_img_bytes
     try: return Image.frombytes(img_mode, img_size, bytes(current_img_bytes))
     except ValueError:
          print("Erreur lors de la reconstruction finale des bytes après itérations.")
          messagebox.showwarning("Glitch Échoué", "La manipulation itérative a rendu l'image invalide.")
          return image_pil

def glitch_bit_plane(image_pil: Image.Image, bit_plane=0, bit_op='zero') -> Image.Image:
    """Glitch de manipulation de plan binaire."""
    if not image_pil: return None
    img = image_pil.copy()
    img_bytes = bytearray(img.tobytes())
    data_len = len(img_bytes)
    if data_len == 0: return img
    bit_plane = max(0, min(int(bit_plane), 7))
    mask = 1 << bit_plane
    if bit_op == 'zero':
        for i in range(data_len): img_bytes[i] = img_bytes[i] & (~mask & 0xFF)
    elif bit_op == 'one':
        for i in range(data_len): img_bytes[i] = img_bytes[i] | mask
    elif bit_op == 'invert':
        for i in range(data_len): img_bytes[i] = img_bytes[i] ^ mask
    try: return Image.frombytes(img.mode, img.size, bytes(img_bytes))
    except ValueError:
        print("Erreur: Les données après manipulation de plan binaire ne peuvent pas être reconstruites.")
        messagebox.showwarning("Glitch Échoué", "La manipulation de plan binaire a rendu l'image invalide.")
        return image_pil

def glitch_advanced_channel_blend(image_pil: Image.Image, channel_a='R', channel_b='G', operation='+') -> Image.Image:
    """Glitch de mélange de canaux avancé."""
    if not image_pil: return None
    img = image_pil.copy()

    r, g, b, a = img.split()
    channels = {'R': r, 'G': g, 'B': b}

    if channel_a not in channels or channel_b not in channels:
        print(f"Canaux sélectionnés invalides: {channel_a}, {channel_b}. Utilisation de R et G par défaut.")
        channel_a, channel_b = 'R', 'G'

    chan_a_img = channels[channel_a]
    chan_b_img = channels[channel_b]

    if operation == 'XOR':
        chan_a_np = np.array(chan_a_img, dtype=np.uint8)
        chan_b_np = np.array(chan_b_img, dtype=np.uint8)
        result_np = np.bitwise_xor(chan_a_np, chan_b_np)
        result_np = result_np.astype(np.uint8)
    else: # '+', '-'
        chan_a_np = np.array(chan_a_img).astype(np.int16)
        chan_b_np = np.array(chan_b_img).astype(np.int16)
        if operation == '+':
            result_np = chan_a_np + chan_b_np
        elif operation == '-':
            result_np = chan_a_np - chan_b_np
        else:
             print(f"Opération '{operation}' inconnue pour mélange canaux avancé. Utilisation de '+'.")
             result_np = chan_a_np + chan_b_np
        result_np = np.clip(result_np, 0, 255).astype(np.uint8)

    channels_rgba_original = [r, g, b, a]
    channel_names = ['R', 'G', 'B', 'A']

    try: channel_a_index = channel_names.index(channel_a)
    except ValueError: channel_a_index = 0

    result_channel_img = Image.fromarray(result_np, 'L')
    channels_rgba_result = list(channels_rgba_original)
    channels_rgba_result[channel_a_index] = result_channel_img

    img_glitched = Image.merge("RGBA", channels_rgba_result)

    return img_glitched

# --- Définition des Glitches et de leurs paramètres pour l'UI ---

GLITCH_DEFINITIONS = {
    "Aucun": { 'func': None, 'params': [], 'tooltip': 'Aucun effet. Réinitialise si l\'image est modifiée.' },
    "Tri de pixels": { 'func': glitch_pixel_sort, 'params': [ {'name': 'threshold', 'label': 'Seuil', 'type': 'slider', 'min': 0, 'max': 255, 'default': 100, 'resolution': 1, 'tooltip': 'Seuil de luminosité/couleur pour débuter un segment à trier.'}, {'name': 'sort_by', 'label': 'Trier par', 'type': 'option', 'options': ['luminosity', 'red', 'green', 'blue'], 'default': 'luminosity', 'tooltip': 'Critère utilisé pour trier les pixels dans les segments.'}, {'name': 'direction', 'label': 'Direction', 'type': 'option', 'options': ['horizontal', 'vertical'], 'default': 'horizontal', 'tooltip': 'Direction du tri des segments.'} ], 'tooltip': 'Trie les pixels dans des segments basés sur un seuil. Crée des barres décalées.' },
    "Destruction de données (bytes)": { 'func': glitch_byte_destruction, 'params': [ {'name': 'num_bytes_to_modify', 'label': 'Nb octets à modifier', 'type': 'slider', 'min': 100, 'max': 500000, 'default': 10000, 'resolution': 1, 'tooltip': 'Nombre d\'octets de données de l\'image à modifier aléatoirement.'}, {'name': 'destruction_type', 'label': 'Type de destruction', 'type': 'option', 'options': ['zero', 'random', 'invert', 'add_random', 'swap'], 'default': 'zero', 'tooltip': 'Méthode de modification des octets : zéro, valeur aléatoire, inverser, ajouter valeur aléatoire, ou échanger des octets.'}, {'name': 'max_byte_change', 'label': 'Changement max (si add_random)', 'type': 'slider', 'min': 1, 'max': 255, 'default': 100, 'resolution': 1, 'tooltip': 'Amplitude maximale du changement ajouté aux octets (pour "add_random").'} ], 'tooltip': 'Modifie directement les octets de données de l\'image. Peut causer une forte corruption.' }, # Correction faute de frappe
    "Mélange de canaux RGB": { 'func': glitch_channel_shuffle, 'params': [], 'tooltip': 'Mélange aléatoirement les canaux de couleur Rouge, Vert et Bleu de l\'image.' },
    "Déchirure horizontale": { 'func': glitch_horizontal_tear, 'params': [ {'name': 'segment_height', 'label': 'Hauteur segment', 'type': 'slider', 'min': 1, 'max': 100, 'default': 10, 'resolution': 1, 'tooltip': 'Hauteur en pixels des segments horizontaux à décaler.'}, {'name': 'max_shift', 'label': 'Décalage max', 'type': 'slider', 'min': 0, 'max': 300, 'default': 50, 'resolution': 1, 'tooltip': 'Décalage horizontal maximal aléatoire pour chaque segment.'}, {'name': 'num_segments', 'label': 'Nb segments (0=tous)', 'type': 'slider', 'min': 0, 'max': 300, 'default': 30, 'resolution': 1, 'tooltip': 'Nombre approximatif de segments à décaler. 0 ou un grand nombre affecte tous les segments.'} ], 'tooltip': 'Crée un effet de "déchirure" en décalant des bandes horizontales de pixels.' },
    "Mélange de blocs (visuel)": { 'func': glitch_block_shuffle, 'params': [ {'name': 'block_size', 'label': 'Taille du bloc', 'type': 'slider', 'min': 5, 'max': 200, 'default': 20, 'resolution': 1, 'tooltip': 'Taille des blocs carrés qui seront échangés.'}, {'name': 'num_shuffles', 'label': 'Nb mélanges', 'type': 'slider', 'min': 10, 'max': 2000, 'default': 100, 'resolution': 1, 'tooltip': 'Nombre d\'échanges aléatoires de paires de blocs à effectuer.'} ], 'tooltip': 'Mélange l\'ordre de blocs carrés de l\'image.' },
    "Mélange de segments de bytes": { 'func': glitch_byte_segment_shuffle, 'params': [ {'name': 'segment_size', 'label': 'Taille segment (bytes)', 'type': 'slider', 'min': 10, 'max': 50000, 'default': 1000, 'resolution': 1, 'tooltip': 'Taille des segments de données (bytes) à mélanger.'}, {'name': 'num_shuffles', 'label': 'Nb mélanges', 'type': 'slider', 'min': 10, 'max': 5000, 'default': 500, 'resolution': 1, 'tooltip': 'Nombre d\'échanges aléatoires de paires de segments de données.'} ], 'tooltip': 'Divise les données de l\'image en segments de bytes et les mélange aléatoirement.' },
    "Corruption par canal RGB": { 'func': glitch_channel_data_bend, 'params': [ {'name': 'num_modifications_per_channel', 'label': 'Nb modif / canal', 'type': 'slider', 'min': 100, 'max': 100000, 'default': 1000, 'resolution': 1, 'tooltip': 'Nombre d\'octets à modifier aléatoirement dans chaque canal de couleur (R, G, B).'}, {'name': 'max_byte_change', 'label': 'Changement max (octet)', 'type': 'slider', 'min': 1, 'max': 255, 'default': 100, 'resolution': 1, 'tooltip': 'Amplitude maximale du changement ajouté aux octets.'} ], 'tooltip': 'Applique une corruption de données indépendamment sur les canaux Rouge, Vert et Bleu.' },
    "Cycle de Teinte (Hue)": { 'func': glitch_hue_cycle, 'params': [ {'name': 'shift_amount', 'label': 'Décalage Teinte', 'type': 'slider', 'min': -255, 'max': 255, 'default': 50, 'resolution': 1, 'tooltip': 'Décale circulairement la teinte de tous les pixels.'} ], 'tooltip': 'Modifie les couleurs en cyclant leur teinte (Hue).' },
     "Miroir / Carrelage": { 'func': glitch_mirror_tile, 'params': [ {'name': 'axis', 'label': 'Axe', 'type': 'option', 'options': ['horizontal', 'vertical'], 'default': 'horizontal', 'tooltip': 'Axe de symétrie et de carrelage.'}, {'name': 'num_repeats', 'label': 'Nb Répétitions', 'type': 'slider', 'min': 1, 'max': 5, 'default': 2, 'resolution': 1, 'tooltip': 'Nombre de fois que l\'image originale/miroir est répétée.'} ], 'tooltip': 'Crée un motif répétitif par duplication et inversion de l\'image.' },
     "Déformation (Warp)": { 'func': glitch_coordinate_warp, 'params': [ {'name': 'warp_intensity', 'label': 'Intensité', 'type': 'slider', 'min': -300, 'max': 300, 'default': 50, 'resolution': 1.0, 'tooltip': 'Force de la distorsion géométrique.'}, {'name': 'frequency', 'label': 'Fréquence Onde', 'type': 'slider', 'min': 0.1, 'max': 20.0, 'default': 1.0, 'resolution': 0.1, 'tooltip': 'Fréquence du motif ondulatoire utilisé pour la déformation.'}, {'name': 'warp_type', 'label': 'Type Onde', 'type': 'option', 'options': ['sin_x', 'sin_y', 'diag_wave', 'radial'], 'default': 'sin_x', 'tooltip': 'Formule mathématique utilisée pour calculer la déformation.'} ], 'tooltip': 'Tord l\'image en déplaçant les pixels selon des motifs ondulatoires.' },
    "Manipulation de Palette": { 'func': glitch_palette_manipulation, 'params': [ {'name': 'num_colors', 'label': 'Nb Couleurs Palette', 'type': 'slider', 'min': 2, 'max': 256, 'default': 256, 'resolution': 1, 'tooltip': 'Limite le nombre de couleurs de l\'image avant de manipuler la palette (Peut dégrader l\'image).'}, {'name': 'manipulation_type', 'label': 'Type manipulation', 'type': 'option', 'options': ['shuffle', 'random_bytes', 'invert_bytes'], 'default': 'shuffle', 'tooltip': 'Méthode de modification des couleurs dans la palette.'}, {'name': 'num_modifications', 'label': 'Nb modif (palette)', 'type': 'slider', 'min': 10, 'max': 768, 'default': 100, 'resolution': 1, 'tooltip': 'Nombre d\'éléments (bytes ou entrées) à modifier dans la palette.'} ], 'tooltip': 'Convertit l\'image en mode palette et modifie la table des couleurs, créant des changements radicaux.' },
    "Lignes de Balayage (Scanlines)": { 'func': glitch_scanlines, 'params': [ {'name': 'intensity', 'label': 'Intensité (0=pas affecté)', 'type': 'slider', 'min': 0, 'max': 255, 'default': 50, 'resolution': 1, 'tooltip': 'Force de l\'effet de scanline.'}, {'name': 'line_height', 'label': 'Hauteur Ligne', 'type': 'slider', 'min': 1, 'max': 20, 'default': 2, 'resolution': 1, 'tooltip': 'Hauteur en pixels des lignes affectées.'}, {'name': 'offset', 'label': 'Décalage Ligne', 'type': 'slider', 'min': 0, 'max': 19, 'default': 0, 'resolution': 1, 'tooltip': 'Décalage vertical pour choisir la première ligne affectée.'}, {'name': 'scanline_color_type', 'label': 'Couleur Ligne', 'type': 'option', 'options': ['darken', 'zero', 'color'], 'default': 'darken', 'tooltip': 'Type d\'effet appliqué aux lignes : assombrir, mettre à zéro (noir), ou appliquer une couleur.'} ], 'tooltip': 'Ajoute des lignes horizontales qui modifient ou assombrissent l\'image, simulant un vieil écran.' },
    "Application Itérative (Simple Shift)": { 'func': glitch_iterative_simple, 'params': [ {'name': 'num_iterations', 'label': 'Nb Itérations', 'type': 'slider', 'min': 1, 'max': 500, 'default': 10, 'resolution': 1, 'tooltip': 'Nombre de fois qu\'un petit décalage de bytes est appliqué. Crée un effet cumulatif.'}, {'name': 'byte_shift_amount', 'label': 'Décalage par étape (bytes)', 'type': 'slider', 'min': -200, 'max': 200, 'default': 5, 'resolution': 1, 'tooltip': 'Nombre de bytes décalés à chaque itération.'} ], 'tooltip': 'Applique un simple glitch (décalage de bytes) plusieurs fois.' },
     "Manipulation Plan Binaire": { 'func': glitch_bit_plane, 'params': [ {'name': 'bit_plane', 'label': 'Plan Binaire (0=moins signif.)', 'type': 'slider', 'min': 0, 'max': 7, 'default': 0, 'resolution': 1, 'tooltip': 'Plan binaire (0 à 7) à modifier dans chaque octet de couleur.'}, {'name': 'bit_op', 'label': 'Opération', 'type': 'option', 'options': ['zero', 'one', 'invert'], 'default': 'zero', 'tooltip': 'Opération binaire appliquée au bit sélectionné : mettre à zéro, mettre à un, ou inverser.'} ] },
     "Mélange Canaux Avancé": {
        'func': glitch_advanced_channel_blend,
        'params': [
            {'name': 'channel_a', 'label': 'Canal A', 'type': 'option', 'options': ['R', 'G', 'B'], 'default': 'R', 'tooltip': 'Premier canal pour l\'opération.'},
            {'name': 'channel_b', 'label': 'Canal B', 'type': 'option', 'options': ['R', 'G', 'B'], 'default': 'G', 'tooltip': 'Deuxième canal pour l\'opération.'},
            {'name': 'operation', 'label': 'Opération', 'type': 'option', 'options': ['+', '-', 'XOR'], 'default': '+', 'tooltip': 'Opération à appliquer entre le Canal A et le Canal B.'}
        ],
        'tooltip': 'Combine deux canaux de couleur (R, G, ou B) en utilisant l\'addition, la soustraction ou l\'opérateur XOR.'
    }
}

# --- Application GUI ---

class GlitchApp:
    def __init__(self, root):
        self.root = root
        root.title("glitchy-py")
        root.geometry("900x700")
        root.option_add('*tearOff', tk.FALSE)

        # --- Configuration du style "Modern" ---
        style = ttk.Style()
        available_themes = style.theme_names()
        if 'clam' in available_themes: style.theme_use('clam')
        elif 'alt' in available_themes: style.theme_use('alt')
        # Sinon, le thème par défaut du système sera utilisé

        # --- Historique Annuler/Rétablir ---
        self.history = []
        self.history_index = -1

        self.image_originale = None
        self.image_actuelle = None
        self.photo_image_tk = None

        # --- Widgets ---
        self.frame_commandes = ttk.Frame(root, padding="10")
        self.frame_commandes.pack(pady=10, padx=10, fill='x')

        # Utilise columnspan=5 pour inclure le bouton Aide
        frame_controles_principaux = ttk.Frame(self.frame_commandes)
        frame_controles_principaux.grid(row=0, column=0, columnspan=5, sticky="ew", pady=(0, 10))

        # Bouton Charger
        btn_charger = ttk.Button(frame_controles_principaux, text="Charger photo", command=self.charger_image)
        btn_charger.pack(side='left', padx=5)

        # Menu Glitch
        ttk.Label(frame_controles_principaux, text="Glitch:").pack(side='left', padx=5)
        self.variable_glitch = tk.StringVar(root)
        self.variable_glitch.set(list(GLITCH_DEFINITIONS.keys())[0])
        self.menu_glitch = ttk.OptionMenu(frame_controles_principaux, self.variable_glitch, self.variable_glitch.get(), *GLITCH_DEFINITIONS.keys(), command=self.update_parameter_ui)
        self.menu_glitch.pack(side='left', padx=5, fill='x', expand=True)

        # Bouton Appliquer
        btn_appliquer = ttk.Button(frame_controles_principaux, text="Appliquer", command=self.appliquer_glitch)
        btn_appliquer.pack(side='left', padx=5)

        # --- Boutons Annuler / Rétablir ---
        self.btn_undo = ttk.Button(frame_controles_principaux, text="Annuler", command=self.undo)
        self.btn_undo.pack(side='left', padx=(20, 2))

        self.btn_redo = ttk.Button(frame_controles_principaux, text="Rétablir", command=self.redo)
        self.btn_redo.pack(side='left', padx=2)

        # Bouton Reset
        btn_reset = ttk.Button(frame_controles_principaux, text="Reset", command=self.reset_image)
        btn_reset.pack(side='left', padx=5)

        # Bouton Enregistrer
        btn_enregistrer = ttk.Button(frame_controles_principaux, text="Enregistrer", command=self.enregistrer_image)
        btn_enregistrer.pack(side='left', padx=5)

        # --- Bouton Aide ---
        btn_help = ttk.Button(frame_controles_principaux, text="Aide", command=self.show_help)
        btn_help.pack(side='left', padx=(20, 5)) # Espacement avant le bouton Aide

        # Frame pour les paramètres dynamiques
        # Utilise columnspan=5
        self.frame_parametres = ttk.LabelFrame(self.frame_commandes, text="Paramètres du Glitch", padding="10")
        self.frame_parametres.grid(row=1, column=0, columnspan=5, sticky="ew")

        # Label pour afficher l'image
        self.label_image = ttk.Label(root, relief="groove")
        self.label_image.pack(pady=10, padx=10)

        # Barre de statut
        self.status_bar = ttk.Label(root, text="Charger une image pour commencer.", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Initialiser l'interface des paramètres
        self.update_parameter_ui()
        # Mettre à jour l'état des boutons
        self.update_button_states()

    # --- Méthode d'aide ---
    def show_help(self):
        """ Affiche une fenêtre d'aide expliquant l'application. """
        help_window = tk.Toplevel(self.root)
        help_window.title("Aide - glitchy")
        help_window.geometry("500x400")
        help_window.transient(self.root) # La fenêtre d'aide reste au-dessus de la principale
        help_window.grab_set() # Capture les événements (la rend semi-modale) - l'utilisateur doit la fermer pour interagir pleinement avec la fenêtre principale
        help_window.resizable(False, False) # Empêche le redimensionnement

        padding_frame = ttk.Frame(help_window, padding="15")
        padding_frame.pack(fill="both", expand=True)

        help_text = """
Bienvenue dans glitchy !

Cette application vous permet d'appliquer divers effets de "glitch" et de
corruption de données sur vos images.

**Comment utiliser :**

1.  Cliquez sur "Charger photo" pour ouvrir un fichier image (PNG, JPG, etc.).
2.  L'image chargée s'affiche au centre. Elle devient l'image "actuelle".
3.  Sélectionnez un "Glitch" dans la liste déroulante.
4.  La section "Paramètres du Glitch" affichera des contrôles (sliders, options)
    spécifiques au glitch sélectionné. Ajustez-les.
5.  Cliquez sur "Appliquer". Le glitch est appliqué à l'image actuelle,
    et le résultat devient la nouvelle image actuelle.
    Vous pouvez appliquer plusieurs glitches successivement.
6.  "Annuler" : Revenir à l'image avant la dernière opération.
7.  "Rétablir" : Appliquer à nouveau l'opération qui vient d'être annulée.
8.  "Reset" : Revenir à l'image originale chargée au début.
9.  "Enregistrer" : Sauvegarder l'image actuelle modifiée dans un fichier.

**Conseils :**

* Passez la souris sur les boutons et les paramètres pour voir des descriptions (Tooltips).
* Certains glitches peuvent rendre l'image illisible ou causer des erreurs.
    Utilisez "Annuler" ou "Reset" si nécessaire.

Amusez-vous à expérimenter !
"""
        # Utiliser un widget Text pour permettre le wrap et potentiellement plus tard du formatage
        # Un Label peut suffire pour du texte simple
        help_label = ttk.Label(padding_frame, text=help_text, wraplength=450, justify="left") # wraplength pour couper les lignes
        help_label.pack(fill="both", expand=True)

        # Bouton pour fermer la fenêtre d'aide
        close_button = ttk.Button(padding_frame, text="Fermer", command=help_window.destroy)
        close_button.pack(pady=10)

        help_window.wait_window() # Attendre que la fenêtre d'aide soit fermée avant de continuer

    # --- Gestion des paramètres dynamiques ---
    def update_parameter_ui(self, *args):
        """ Met à jour les widgets de paramètres en fonction du glitch sélectionné. """
        for widget in self.frame_parametres.winfo_children():
            widget.destroy()
        self.parameter_widgets = {}

        selected_glitch_name = self.variable_glitch.get()
        glitch_info = GLITCH_DEFINITIONS.get(selected_glitch_name)

        if not glitch_info or not glitch_info['params']:
            ttk.Label(self.frame_parametres, text="Aucun paramètre pour ce glitch.").pack(padx=5, pady=5)
            return

        row_idx = 0
        for param_info in glitch_info['params']:
            frame_param = ttk.Frame(self.frame_parametres)
            frame_param.grid(row=row_idx, column=0, sticky="ew", pady=2)

            label_text = param_info.get('label', param_info['name'])
            param_label_widget = ttk.Label(frame_param, text=label_text + ":")
            param_label_widget.pack(side='left', padx=(0, 5))

            if param_info['type'] == 'slider':
                resolution = param_info.get('resolution', 1)
                if isinstance(param_info.get('default', 0), int) and resolution == 1:
                     scale_var = tk.IntVar(value=param_info['default'])
                else:
                     scale_var = tk.DoubleVar(value=param_info['default'])
                     if resolution == 1: resolution = 1.0

                slider = tk.Scale(frame_param, from_=param_info['min'], to=param_info['max'],
                                  orient=HORIZONTAL, variable=scale_var,
                                  length=250,
                                  resolution=resolution,
                                  command=self.update_status_param_preview
                                  )

                slider.pack(side='left', fill='x', expand=True)
                self.parameter_widgets[param_info['name']] = scale_var

            elif param_info['type'] == 'option':
                option_var = tk.StringVar(value=param_info['default'])
                option_menu = ttk.OptionMenu(frame_param, option_var, option_var.get(), *param_info['options'], command=self.update_status_param_preview)
                option_menu.pack(side='left', padx=5, fill='x', expand=True)
                self.parameter_widgets[param_info['name']] = option_var

            row_idx += 1

        self.frame_parametres.grid_columnconfigure(0, weight=1)

    def get_glitch_parameters(self):
        """ Récupère les valeurs actuelles des paramètres. """
        params = {}
        selected_glitch_name = self.variable_glitch.get()
        glitch_info = GLITCH_DEFINITIONS.get(selected_glitch_name)
        if not glitch_info: return params
        for param_info in glitch_info['params']:
            param_name = param_info['name']
            widget_var = self.parameter_widgets.get(param_name)
            if widget_var:
                value = widget_var.get()
                if param_info['type'] == 'slider':
                     resolution = param_info.get('resolution', 1)
                     if resolution == 1 and isinstance(value, float) and value.is_integer():
                         params[param_name] = int(value)
                     else: params[param_name] = value
                else: params[param_name] = value
        return params

    def update_status(self, message):
        """ Met à jour le texte de la barre de statut. """
        self.status_bar.config(text=message)

    def update_status_param_preview(self, value=None):
        """ Met à jour le statut avec un aperçu du paramètre en cours d'ajustement. """
        selected_glitch_name = self.variable_glitch.get()
        glitch_info = GLITCH_DEFINITIONS.get(selected_glitch_name)
        if not glitch_info or not glitch_info['params']:
             self.update_status(f"Paramètres pour: {selected_glitch_name}")
             return
        param_label_preview = ""
        param_found = False
        for param_info in glitch_info['params']:
             widget_var = self.parameter_widgets.get(param_info['name'])
             if widget_var:
                 current_value = widget_var.get()
                 if param_info['type'] == 'slider' and current_value == value:
                      param_label_preview = param_info.get('label', param_info['name'])
                      param_found = True; break
                 elif param_info['type'] == 'option' and isinstance(value, str) and current_value == value:
                      param_label_preview = param_info.get('label', param_info['name'])
                      param_found = True; break
        if param_found:
             if isinstance(value, float): display_value = f"{value:.2f}"
             else: display_value = value
             self.update_status(f"{selected_glitch_name} - {param_label_preview}: {display_value}")
        else: self.update_status(f"Paramètres pour: {selected_glitch_name}")

    def update_button_states(self):
        """ Active ou désactive les boutons (y compris Annuler/Rétablir) selon l'état de l'application. """
        is_image_loaded = self.image_actuelle is not None
        can_undo = self.history_index > 0
        can_redo = self.history_index < len(self.history) - 1

        # Boutons principaux (Charger est toujours actif)
        for widget in self.frame_commandes.winfo_children():
            if isinstance(widget, ttk.Frame):
                for btn in widget.winfo_children():
                     if isinstance(btn, ttk.Button):
                         btn_text = btn.cget('text')
                         if btn_text == "Appliquer":
                              btn.config(state=tk.NORMAL if is_image_loaded else tk.DISABLED)
                         elif btn_text == "Reset":
                              btn.config(state=tk.NORMAL if self.history_index > 0 or self.image_originale else tk.DISABLED)
                         elif btn_text == "Enregistrer":
                              btn.config(state=tk.NORMAL if is_image_loaded else tk.DISABLED)
                         # Gestion Annuler/Rétablir
                         elif btn_text == "Annuler":
                             btn.config(state=tk.NORMAL if can_undo else tk.DISABLED)
                         elif btn_text == "Rétablir":
                             btn.config(state=tk.NORMAL if can_redo else tk.DISABLED)
                         # Bouton Aide est toujours actif
                         elif btn_text == "Aide":
                             btn.config(state=tk.NORMAL)
                break

    # --- Gestion de l'historique (Annuler/Rétablir) ---

    def add_history_step(self, image_pil: Image.Image):
        """ Ajoute l'image actuelle à l'historique et gère la taille/index. """
        if image_pil is None: return

        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]

        self.history.append(image_pil.copy())

        if len(self.history) > MAX_HISTORY_SIZE:
            excess_count = len(self.history) - MAX_HISTORY_SIZE
            self.history = self.history[excess_count:]
            self.history_index = MAX_HISTORY_SIZE - 1
        else:
             self.history_index += 1

        self.update_button_states()

    def undo(self):
        """ Annule la dernière opération. """
        if self.history_index > 0:
            self.history_index -= 1
            self.image_actuelle = self.history[self.history_index].copy()
            self.afficher_image(self.image_actuelle)
            self.update_button_states()
            self.update_status(f"Annulé. Étape {self.history_index + 1}/{len(self.history)}.")

    def redo(self):
        """ Rétablit l'opération annulée. """
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.image_actuelle = self.history[self.history_index].copy()
            self.afficher_image(self.image_actuelle)
            self.update_button_states()
            self.update_status(f"Rétabli. Étape {self.history_index + 1}/{len(self.history)}.")
        else:
            self.update_status("Impossible de rétablir davantage.")

    # --- Gestion de l'image et des actions ---

    def charger_image(self):
        file_path = filedialog.askopenfilename( initialdir=os.getcwd(), title="Sélectionner un fichier image", filetypes=IMAGE_FILETYPES )
        if file_path:
            try:
                self.update_status(f"Chargement de {os.path.basename(file_path)}...")
                with Image.open(file_path) as img_pil:
                    self.image_originale = ensure_rgb_or_rgba(img_pil.copy())
                    self.image_actuelle = self.image_originale.copy()

                # Initialiser l'historique
                self.history = []
                self.history_index = -1
                self.add_history_step(self.image_actuelle)

                self.afficher_image(self.image_actuelle)
                self.variable_glitch.set(list(GLITCH_DEFINITIONS.keys())[0])
                self.update_parameter_ui()
                self.update_button_states()
                self.update_status(f"'{os.path.basename(file_path)}' chargé. Prêt à glitcher.")
            except Exception as e:
                self.update_status(f"Erreur de chargement.")
                messagebox.showerror("Erreur de chargement", f"Impossible de charger l'image: {e}")
                self.reset_image()

    def afficher_image(self, image_pil: Image.Image):
        if not image_pil:
            self.label_image.config(image="")
            self.photo_image_tk = None
            return
        img_pil_display = image_pil.copy()
        img_pil_display.thumbnail(MAX_DISPLAY_SIZE, Resampling)
        try:
             if img_pil_display.mode not in ['RGB', 'RGBA']: img_pil_display = img_pil_display.convert('RGBA')
             self.photo_image_tk = ImageTk.PhotoImage(img_pil_display)
             self.label_image.config(image=self.photo_image_tk)
             self.label_image.image = self.photo_image_tk
        except Exception as e:
             print(f"Erreur lors de l'affichage de l'image : {e}")
             import traceback; traceback.print_exc()
             self.label_image.config(image="")
             self.photo_image_tk = None
             messagebox.showwarning("Erreur Affichage", "Impossible d'afficher l'image modifiée.")

    def appliquer_glitch(self):
        if not self.image_actuelle:
            messagebox.showwarning("Attention", "Aucune image chargée.")
            self.update_status("Impossible d'appliquer : aucune image chargée.")
            return

        selected_glitch_name = self.variable_glitch.get()
        glitch_info = GLITCH_DEFINITIONS.get(selected_glitch_name)

        if not glitch_info or glitch_info['func'] is None:
             if self.image_originale:
                 self.image_actuelle = self.image_originale.copy()
                 self.afficher_image(self.image_actuelle)
                 # Réinitialiser l'historique à l'original si on sélectionne "Aucun"
                 self.history = [self.image_originale.copy()]
                 self.history_index = 0
                 self.update_button_states()
                 self.update_status("Image réinitialisée à l'original via 'Aucun'.")
             return

        params = self.get_glitch_parameters()
        self.update_status(f"Application de '{selected_glitch_name}'...")

        try:
            img_glitched = glitch_info['func'](self.image_actuelle.copy(), **params)

            if img_glitched and isinstance(img_glitched, Image.Image):
                self.image_actuelle = img_glitched
                self.afficher_image(self.image_actuelle)
                self.add_history_step(self.image_actuelle)
                self.update_status(f"'{selected_glitch_name}' appliqué avec succès.")
            else:
                 self.update_status(f"'{selected_glitch_name}' a échoué ou n'a pas retourné d'image valide.")
                 messagebox.showwarning("Glitch Échoué", f"Le glitch '{selected_glitch_name}' n'a pas retourné une image valide.")

        except Exception as e:
            self.update_status(f"Erreur lors de l'application de '{selected_glitch_name}'.")
            messagebox.showerror("Erreur Glitch", f"Une erreur s'est produite lors de l'application du glitch: {e}")
            import traceback; traceback.print_exc()

    def reset_image(self):
        """ Revenir à l'image originale chargée et réinitialiser l'historique. """
        if self.image_originale:
            self.image_actuelle = self.image_originale.copy()
            self.afficher_image(self.image_actuelle)
            # Réinitialiser l'historique
            self.history = [self.image_originale.copy()]
            self.history_index = 0
            self.variable_glitch.set(list(GLITCH_DEFINITIONS.keys())[0])
            self.update_parameter_ui()
            self.update_button_states()
            self.update_status("Image réinitialisée à l'original.")
        else:
            self.image_actuelle = None; self.image_originale = None; self.afficher_image(None)
            self.history = []
            self.history_index = -1
            self.update_button_states()
            self.update_status("Aucune image chargée.")

    def enregistrer_image(self, ):
        """ Enregistre l'image actuellement affichée dans un fichier. """
        if not self.image_actuelle:
            messagebox.showwarning("Attention", "Aucune image à enregistrer.")
            self.update_status("Impossible d'enregistrer : aucune image chargée.")
            return
        file_path = filedialog.asksaveasfilename( initialdir=os.getcwd(), title="Enregistrer l'image glitchée", defaultextension=DEFAULT_SAVE_EXTENSION, filetypes=IMAGE_FILETYPES )
        if file_path:
            try:
                self.update_status(f"Enregistrement de '{os.path.basename(file_path)}'...")
                img_to_save = self.image_actuelle.copy()
                if img_to_save.mode not in ['RGB', 'RGBA']: img_to_save = img_to_save.convert('RGBA')
                img_to_save.save(file_path)
                self.update_status(f"Image enregistrée sous '{os.path.basename(file_path)}'.")
                messagebox.showinfo("Succès", "Image enregistrée avec succès!")
            except Exception as e:
                self.update_status(f"Erreur lors de l'enregistrement.")
                messagebox.showerror("Erreur d'enregistrement", f"Impossible d'enregistrer l'image: {e}")
                import traceback; traceback.print_exc()

# --- Lancement de l'application ---
if __name__ == "__main__":
    root = tk.Tk()
    app = GlitchApp(root)
    root.mainloop()
