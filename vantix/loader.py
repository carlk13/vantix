import torch
import numpy as np
from torch.utils.data import IterableDataset
from ._vantix import load_images_fast  # Import directly from the Rust binary

class vantixLoader(IterableDataset):
    """
    High-performance Rust-backed image loader.
    """
    def __init__(self, image_paths, width=224, height=224, batch_size=64, augment=False, shuffle=True):
        self.image_paths = np.array(image_paths)
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle

        self.pin = torch.cuda.is_available()
        
    def __iter__(self):
        indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(indices)
            
        for start_idx in range(0, len(indices), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            batch_paths = self.image_paths[batch_indices].tolist()
            
            # Call Rust
            # returns: (B, C, H, W) float32 array
            np_batch = load_images_fast(batch_paths, self.width, self.height, self.augment)
            
            # Zero-copy conversion to Torch
            tensor = torch.from_numpy(np_batch)
            if self.pin:
                tensor = tensor.pin_memory()
                
            yield tensor

    def __len__(self):
        return (len(self.image_paths) + self.batch_size - 1) // self.batch_size