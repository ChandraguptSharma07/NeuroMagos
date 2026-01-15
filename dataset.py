import torch, glob, os, pandas as pd, numpy as np
from torch.utils.data import Dataset
import preprocessing as pp

class SynapseDataset(Dataset):
    def __init__(self, root, mode='train'):
        self.files = glob.glob(os.path.join(root, "**", "gesture*.csv"), recursive=True)
        self.mode = mode
        if not self.files: print(f"warn: no files in {root}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        
        # label parsing: gesture00 -> 0
        try:
            lbl = int(os.path.basename(f).split('_')[0].replace('gesture', ''))
        except:
            lbl = -1

        try:
            # load & prep
            raw = pd.read_csv(f).iloc[:, :8].values
            
            d = pp.notch(raw)
            d = pp.bandpass(d)
            d = pp.norm(d)
            
            # spectrogram [8, 33, 81]
            spec = torch.tensor(pp.get_spec(d), dtype=torch.float32)

            # spec augment (train only)
            if self.mode == 'train':
                if np.random.rand() < 0.5: # freq mask
                    fm = np.random.randint(0, 5)
                    fs = np.random.randint(0, 33 - fm)
                    spec[:, fs:fs+fm, :] = 0
                
                if np.random.rand() < 0.5: # time mask
                    tm = np.random.randint(0, 10)
                    ts = np.random.randint(0, 81 - tm)
                    spec[:, :, ts:ts+tm] = 0

            return spec, torch.tensor(lbl, dtype=torch.long)
            
        except Exception as e:
            print(f"err: {f} - {e}")
            return torch.zeros(1), torch.tensor(-1)
