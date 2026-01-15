import torch
import torch.nn as nn
from torchvision import models

class HybridNeuroMagos(nn.Module): 
    def __init__(self, num_channels=8, num_classes=5):
        super().__init__()
        
        # resnet18 from scratch
        self.rn = models.resnet18(weights=None)
        
        # mod first layer to accept 8 channels
        c1 = self.rn.conv1
        self.rn.conv1 = nn.Conv2d(num_channels, c1.out_channels, 
                                  kernel_size=c1.kernel_size, stride=c1.stride, 
                                  padding=c1.padding, bias=False)
        
        # mod last layer
        self.rn.fc = nn.Linear(self.rn.fc.in_features, num_classes)

    def forward(self, x):
        return self.rn(x)
