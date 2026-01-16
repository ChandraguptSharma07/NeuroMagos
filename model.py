import torch, torch.nn as nn
from torchvision import models

class HybridNeuroMagos(nn.Module): 
    def __init__(self, c=8, k=5):
        super().__init__()
        # vanilla resnet
        self.rn = models.resnet18(weights=None)
        
        # 8-channel input mod
        w = self.rn.conv1
        self.rn.conv1 = nn.Conv2d(c, w.out_channels, kernel_size=w.kernel_size, 
                                  stride=w.stride, padding=w.padding, bias=False)
        
        # dropout + fc
        self.rn.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.rn.fc.in_features, k))

    def forward(self, x): return self.rn(x)

class CNN1D(nn.Module):
    def __init__(self, c=8, k=5):
        super().__init__()
        
        def blk(i, o, k, s=1):
            return nn.Sequential(
                nn.Conv1d(i, o, k, stride=s, padding=k//2),
                nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2)
            )
            
        self.net = nn.Sequential(
            blk(c, 64, 7, 2), # 750
            blk(64, 128, 5),  # 375
            blk(128, 256, 3), # 187
            blk(256, 512, 3), # 93
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, k)
        )

    def forward(self, x): return self.net(x)
