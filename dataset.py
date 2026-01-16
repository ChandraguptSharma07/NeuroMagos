import torch, glob, os, pandas as pd, numpy as np
from torch.utils.data import Dataset
import preprocessing as pp

class SynapseDataset(Dataset):
    def __init__(self, root, mode='train', type='spec'):
        self.f = glob.glob(f"{root}/**/gesture*.csv", recursive=True)
        self.mode, self.type = mode, type
        if not self.f: print(f"warn: empty {root}")

    def __len__(self): return len(self.f)

    def __getitem__(self, idx):
        path = self.f[idx]
        try:
            # gesture00 -> 0
            lbl = int(os.path.basename(path).split('_')[0].replace('gesture', ''))
        except: lbl = -1

        try:
            raw = pd.read_csv(path).iloc[:, :8].values
            d = pp.norm(pp.bandpass(pp.notch(raw)))
            
            if self.type == 'raw':
                # pad/crop to 3000
                if d.shape[0] < 3000:
                    d = np.pad(d, ((0, 3000-d.shape[0]), (0, 0)))
                else: d = d[:3000]
                t = torch.tensor(d.T, dtype=torch.float32)
            else:
                s = pp.get_spec(d) # [8, 33, 81]
                t = torch.tensor(s, dtype=torch.float32)

                if self.mode == 'train':
                    # spec augment
                    if np.random.rand() < 0.5: # freq
                        f = np.random.randint(0, 5)
                        fs = np.random.randint(0, 33-f)
                        t[:, fs:fs+f, :] = 0
                    if np.random.rand() < 0.5: # time
                        tm = np.random.randint(0, 10)
                        ts = np.random.randint(0, 81-tm)
                        t[:, :, ts:ts+tm] = 0

            return t, torch.tensor(lbl, dtype=torch.long)
            
        except: return torch.zeros(1), torch.tensor(-1)
