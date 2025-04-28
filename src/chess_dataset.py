#overrides pytorch dataset class that loads HDF5 files

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import time

class ChessDataset(Dataset):
    def __init__(self, file_path, fraction=1.0):
        self.file_path = file_path
        start_time = time.time()
        
        with h5py.File(file_path, "r") as f:
            total_size = len(f["X"])
            subset_size = int(total_size * fraction)

            print(f"Total dataset size: {total_size}, Loading fraction: {fraction} ({subset_size} samples)")
            #read the entire dataset in large sequential chunks
            x_shape = f["X"].shape
            y_shape = f["Y"].shape
            
            #calculated chunks needed
            chunk_size = 50000
            num_chunks = (total_size + chunk_size - 1) // chunk_size
            
            # First, create a boolean mask for the indices we want to keep
            keep_mask = np.zeros(total_size, dtype=bool)
            keep_indices = np.random.choice(total_size, subset_size, replace=False)
            keep_mask[keep_indices] = True
            
            # Preallocate arrays for the data we'll keep
            self.data = np.empty((subset_size,) + tuple(x_shape[1:]), dtype=np.float32)
            self.labels = np.empty((subset_size,) + tuple(y_shape[1:]), dtype=np.float32)
            
            # Process the data in large sequential chunks
            loaded_samples = 0
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_size)
                
                # Read a large chunk sequentially
                print(f"Reading chunk {chunk_idx+1}/{num_chunks} from disk...")
                x_chunk = f["X"][start_idx:end_idx]
                y_chunk = f["Y"][start_idx:end_idx]
                
                # Find which samples from this chunk we want to keep
                chunk_mask = keep_mask[start_idx:end_idx]
                chunk_samples = np.sum(chunk_mask)
                
                if chunk_samples > 0:
                    # Copy only the samples we want to keep
                    self.data[loaded_samples:loaded_samples+chunk_samples] = x_chunk[chunk_mask].astype(np.float32)
                    self.labels[loaded_samples:loaded_samples+chunk_samples] = y_chunk[chunk_mask].astype(np.float32)
                    loaded_samples += chunk_samples
                
                print(f"Processed {end_idx}/{total_size} samples, kept {loaded_samples}/{subset_size}")

            elapsed = time.time() - start_time
            print(f"✅ Finished loading dataset into RAM in {elapsed:.2f} seconds.")

        self.length = len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])
        y = torch.from_numpy(self.labels[idx])
        return x, y

    def __len__(self):
        return self.length


