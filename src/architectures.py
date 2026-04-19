import torch
import torch.nn as nn
import torch.nn.functional as F

# architecture for the 3D CNNs
class ParkinsonClassifier3D(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ParkinsonClassifier3D, self).__init__()
        
        # Layer 1: Conv -> Batch Norm -> ReLU -> Pool
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        
        # Layer 2: Conv -> Batch Norm -> ReLU -> Pool
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        
        # Layer 3: Conv -> Batch Norm -> ReLU -> Pool
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        
        self.pool = nn.MaxPool3d(2)
        self.gap = nn.AdaptiveAvgPool3d(1)
        
        # Fully Connected Layers with Dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # We follow the pattern: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.gap(x)
        x = x.view(-1, 64)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # a tenir en compte: using RAW LOGITS. 
        # posar nn.BCEWithLogitsLoss() a training
        return self.fc2(x)
    

# architecture for the 2D models, same as the 3D
class ParkinsonClassifier2D(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ParkinsonClassifier2D, self).__init__()
        
        # Changed to 2D
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.gap(x)
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)