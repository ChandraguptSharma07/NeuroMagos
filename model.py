import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridNeuroMagos(nn.Module):
    def __init__(self, num_channels=8, num_classes=5):
        super(HybridNeuroMagos, self).__init__()
        
        # Dual Branch CNN: 
        # Branch 1: High freq (small kernel)
        self.b1_conv1 = nn.Conv1d(num_channels, 32, kernel_size=3, padding=1)
        self.b1_bn1 = nn.BatchNorm1d(32)
        self.b1_conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.b1_bn2 = nn.BatchNorm1d(64)

        # Branch 2: Low freq (large kernel)
        self.b2_conv1 = nn.Conv1d(num_channels, 32, kernel_size=11, padding=5)
        self.b2_bn1 = nn.BatchNorm1d(32)
        self.b2_conv2 = nn.Conv1d(32, 64, kernel_size=11, padding=5)
        self.b2_bn2 = nn.BatchNorm1d(64)

        self.pool = nn.MaxPool1d(4) 
        
        # LSTM part
        # concatenated features size = 64 + 64 = 128
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [Batch, 8, seq_len]
        
        # Branch 1
        x1 = F.relu(self.b1_bn1(self.b1_conv1(x)))
        x1 = F.relu(self.b1_bn2(self.b1_conv2(x1)))
        x1 = self.pool(x1)
        
        # Branch 2
        x2 = F.relu(self.b2_bn1(self.b2_conv1(x)))
        x2 = F.relu(self.b2_bn2(self.b2_conv2(x2)))
        x2 = self.pool(x2)
        
        # Combine
        x_cat = torch.cat((x1, x2), dim=1) # [Batch, 128, reduced_len]
        
        # Permute for LSTM [Batch, Len, Feat]
        x_lstm = x_cat.permute(0, 2, 1)
        
        # Run LSTM
        # out: [Batch, Len, Hidden*2]
        out, _ = self.lstm(x_lstm)
        
        # Take last time step
        last_out = out[:, -1, :] 
        
        return self.fc(last_out)

if __name__ == "__main__":
    model = HybridNeuroMagos()
    dummy = torch.randn(2, 8, 2560)
    print("Output shape:", model(dummy).shape)
