import os
import re
import numpy as np
import tifffile

def natural_sort_key(s):
    """Sorts strings containing numbers logically (e.g., 'frame_2' before 'frame_10')."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

class LazyTiffSequence:
    def __init__(self, folder_path, buffer_radius=20):
        self.folder_path = folder_path
        self.buffer_radius = buffer_radius
        
        # Robust extension check (.TIF, .tiff, .tif, etc.)
        valid_ext = (".tif", ".tiff")
        all_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_ext)]
        
        if not all_files:
            raise ValueError(f"No TIFF files found in {folder_path}")
            
        # Natural sort ensures sequence order is perfect regardless of naming convention
        self.files = sorted(all_files, key=natural_sort_key)
        self.num_frames = len(self.files)
        
        # This dictionary is our sliding RAM buffer
        self.cache = {} 

    def get_index_from_filename(self, filename):
        """Finds where a specifically chosen file sits in the sequence."""
        try:
            return self.files.index(filename)
        except ValueError:
            return 0

    def prefetch_window(self, center_index):
        """Pre-loads the initial window so the first few scrubs are instantly smooth."""
        min_fetch = max(0, center_index - self.buffer_radius)
        max_fetch = min(self.num_frames - 1, center_index + self.buffer_radius)
        
        print(f"Pre-caching frames {min_fetch + 1} to {max_fetch + 1}...")
        for i in range(min_fetch, max_fetch + 1):
            if i not in self.cache:
                file_path = os.path.join(self.folder_path, self.files[i])
                self.cache[i] = tifffile.imread(file_path)

    def get_frame(self, index):
        """Returns the frame, dynamically managing the sliding memory window."""
        if index < 0 or index >= self.num_frames:
            return None
            
        # 1. Cache Miss: Load the requested frame if it isn't in RAM
        if index not in self.cache:
            file_path = os.path.join(self.folder_path, self.files[index])
            self.cache[index] = tifffile.imread(file_path)
            
        # 2. Garbage Collection: Determine our allowed +/- window
        min_keep = max(0, index - self.buffer_radius)
        max_keep = min(self.num_frames - 1, index + self.buffer_radius)
        
        # 3. Purge anything outside that window from RAM
        # Use list() to avoid dictionary "changed size during iteration" errors
        keys_to_delete = [k for k in list(self.cache.keys()) if k < min_keep or k > max_keep]
        for k in keys_to_delete:
            del self.cache[k]
            
        return self.cache[index]

def build_display_lut(vmin, vmax, gamma, dtype_max=65535):
    """Precompute a uint8 LUT for uint16 -> display uint8 mapping.

    Returns a (dtype_max+1,) uint8 array. Per-frame display becomes lut[frame],
    a single indexed pass versus 4 full-array allocations in the naive path.
    """
    span = float(vmax - vmin)
    if span <= 0:
        span = 1e-8
    indices = np.arange(dtype_max + 1, dtype=np.float32)
    np.clip(indices, vmin, vmax, out=indices)
    indices -= vmin
    indices /= span
    if gamma != 1.0:
        np.power(indices, gamma, out=indices)
    indices *= 255.0
    return indices.astype(np.uint8)


def scale_16bit_to_8bit(image_array, vmin, vmax, gamma=1.0, lut=None):
    """Scale an image array to uint8 for display.

    When lut is provided and frame dtype is uint16, uses a single indexed pass
    (lut[frame]) instead of 4 full-array allocations. Build lut once with
    build_display_lut() and pass it here on every frame.
    """
    if lut is not None and image_array.dtype == np.uint16:
        return lut[image_array]

    clipped = np.clip(image_array, vmin, vmax)
    span = vmax - vmin
    if span <= 0:
        span = 1e-8
    normalized = (clipped - vmin) / span
    if gamma != 1.0:
        normalized = np.power(normalized, gamma)
    return (normalized * 255.0).astype(np.uint8)

