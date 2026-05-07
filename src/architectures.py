import torch
import torch.nn as nn
import torch.nn.functional as F

# architecture for the CNNs


class ParkinsonClassifier3D(nn.Module):
    """
    Small 3d (custom, trained from scratch)
    3D 3-layer CNN
    Input: (B, 1, H, W, D)
    Output: (B, 1) raw logit (needs BCEWithLogitsLoss during trainin)
    """
    def __init__(self, dropout_rate=0.3):

        #super(ParkinsonClassifier3D, self).__init__()
        super().__init__()
        
        # Layer 1: Conv -> Batch Norm -> ReLU -> Pool
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm3d(16)
        
        # Layer 2: Conv -> Batch Norm -> ReLU -> Pool
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm3d(32)
        
        # Layer 3: Conv -> Batch Norm -> ReLU -> Pool
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm3d(64)
        
        self.pool = nn.MaxPool3d(2)
        self.gap  = nn.AdaptiveAvgPool3d(1)
        
        # Fully Connected Layers with Dropout
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
        
        # take into account: using RAW LOGITS. 
        # nn.BCEWithLogitsLoss()
        return self.fc2(x)



# Deeper 3d
# (custom, trained from scratch)
class ParkinsonClassifier3D_deeper(nn.Module):
    """
    4-layer 3D-CNN (16->32->64->128 filters).
    Input : (B, 1, H, W, D)
    Output: (B, 1)  raw logit (same as before, cuidadu with BCEWithLogitsLoss when training)
    """
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16),  nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(), nn.MaxPool3d(2),
        )
        self.gap     = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1     = nn.Linear(128, 64)
        self.fc2     = nn.Linear(64, 1)
 
    def forward(self, x):
        x = self.gap(self.features(x)).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
 

# architecture for the 2D models

#  same as the small 3D
class ParkinsonClassifier2D(nn.Module):
    """
    Small 2d (custom, trained from scratch)
    2D 3-layer CNN
    Input: (B, 1, H, W) -- the projection / sum of slices
    Output: (B, 1) raw logit
    """
    def __init__(self, dropout_rate=0.3):
        #super(ParkinsonClassifier2D, self).__init__()
        super().__init__()
        
        # Changed to 2D from the 3D architecture
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
    

#  2.5D ResNet18 (pretrained on ImageNet)
# Why this is worth trying
# Training a 3D-CNN from scratch needs a lot of data to converge well.
# Reusing `ImageNe`t weights gives the network a strong visual prior from
# the start: edges, textures, shapes: all useful even for medical images.
# The 2.5D trick lets us exploit those 2D weights on volumetric data.

import torchvision.models as tv_models
 
class ParkinsonClassifier25D(nn.Module):
    """
    ResNet18 pretrained on ImageNet, fine-tuned for binary PD classification.
 
    Input : (B, 3, H, W)
            The 3 channels are the axial, coronal and sagittal central slices
            produced by get_25d_transforms(). They look like an RGB image to
            the backbone, but each channel carries a different spatial view
            of the DaTSCAN volume.
 
    Output: (B, 1)  raw logit — use BCEWithLogitsLoss.
    """
    def __init__(self, dropout_rate=0.3, pretrained=True):
        super().__init__()
        weights  = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
 
        # Strip the original ImageNet classification head
        # Everything up to (and including) the global avg pool is kept.
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        # backbone.children() ends with: avgpool -> (B,512,1,1)
        # then fc -> (B,1000)  we replace this
 
        self.dropout = nn.Dropout(dropout_rate)
        self.fc      = nn.Linear(512, 1)
 
    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.features(x)       # -> (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # -> (B, 512)
        x = self.dropout(x)
        return self.fc(x)          # -> (B, 1)
 
