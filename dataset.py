import torch
from torch.utils.data import Dataset
import glob
import os
import pandas as pd
import numpy as np
import preprocessing

class SynapseDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        """
        Args:
            root_dir (string): Directory with all the csv files.
            mode (string): 'train' or 'test' (if we had a split).
        """
        self.root_dir = root_dir
        # Recursive glob to look for all 'gesture*.csv' files in all subfolders
        self.file_list = glob.glob(os.path.join(root_dir, "**", "gesture*.csv"), recursive=True)
        
        # Simple sanity check
        if len(self.file_list) == 0:
            print(f"Warning: No files found in {root_dir}")
        else:
            print(f"Found {len(self.file_list)} samples.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        # 1. Parse Label
        # Filename example: gesture00_trial01.csv
        filename = os.path.basename(file_path)
        try:
            # "gesture00" -> 0
            label_str = filename.split('_')[0].replace('gesture', '')
            label = int(label_str)
        except ValueError:
            print(f"Error parsing label from {filename}")
            label = -1 # Indicate error

        # 2. Load Data
        try:
            # Skip header if it exists (it does), pandas handles it by default
            df = pd.read_csv(file_path)
            # Take first 8 columns (channels)
            raw_data = df.iloc[:, :8].values
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return torch.zeros(1), torch.tensor(-1)

        # 3. Preprocess
        # Apply the pipeline we verified in utils.py
        # Notch -> Bandpass -> Normalize
        filtered = preprocessing.apply_notch_filter(raw_data)
        filtered = preprocessing.apply_bandpass_filter(filtered)
        normalized = preprocessing.normalize_data(filtered)
        
        # 4. Convert to Tensor
        # Shape: [Time, Channels] -> PyTorch usually expects [Channels, Time] for Conv1d
        # Let's keep it [Time, Channels] for now and transpose if needed by the model.
        # But wait, Conv1d needs (Batch, Channels, Length).
        # Let's transpose it now to be safe: (8, 2560)
        signal_tensor = torch.tensor(normalized, dtype=torch.float32).transpose(0, 1)
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return signal_tensor, label_tensor

# Quick verify block
if __name__ == "__main__":
    # Test on the current directory structure
    dataset = SynapseDataset(root_dir="Synapse_Dataset")
    
    if len(dataset) > 0:
        sig, lbl = dataset[0]
        print(f"Signal Shape: {sig.shape}") # Should be [8, ~2560]
        print(f"Label: {lbl}")
        print(f"Signal Mean: {sig.mean():.4f} (should be ~0)")
        print(f"Signal Std: {sig.std():.4f} (should be ~1)")
