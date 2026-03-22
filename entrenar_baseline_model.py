# %% [markdown]
# # Try to use pytorch on the PPMI data
# 
# First import libs and define paths

# %%
import pandas as pd
import os

# %%

base_path = "/home/data/PPMI"
participants_file = os.path.join(base_path, "rawdata/participants.tsv")
curated_file = os.path.join(base_path, "documents/PPMI_Curated_Data_Cut_Public_20240729.xlsx")

# %% [markdown]
# # Pytorch
# 
# If all above works, load `pytorch` and `nibabel`.

# %%
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np

# %% [markdown]
# pytorch needs a dataset class to handle the "fetching" of stuff like which patient is and is not sick.

# %%
class PPMIDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load the 3D NIfTI file
        img = nib.load(self.file_paths[idx]).get_fdata()

        # Preprocessing
        img = (img - np.mean(img)) / np.std(img) # Normalize intensity

        # Convert to Torch Tensor (Add a 'channel' dimension for the CNN)
        # Shape becomes: [1, Depth, Height, Width]
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)

        label = torch.tensor(self.labels[idx])
        return img_tensor, label

# %%
import glob

# %%
derivatives_path = os.path.join(base_path, "derivatives/dat-reg-v6")
excel_path = os.path.join(base_path, "documents/PPMI_Curated_Data_Cut_Public_20240729.xlsx")

# Load the labels
df = pd.read_excel(excel_path)

# agafam només el num del pacient i el seu cohort
labels_map = df[['PATNO', 'COHORT']].drop_duplicates().set_index('PATNO')['COHORT'].to_dict()
#print("labels map: ", labels_map)

# trobam totes les "ses-BL" de DaTscan
baseline_images = glob.glob(f"{derivatives_path}/sub-*/ses-BL/spect/*_DaTSCAN.nii.gz") # màgia negra
#print("baseline_images: ", baseline_images)

data_list = []

for img_path in baseline_images:

    # agafam PATNO com a int del filename (aka sub-PPMI100001 -> 100001)
    sub_id = img_path.split('/')[-4] # només 'sub-PPMI100001'
    patno = int(sub_id.replace('sub-PPMI', ''))

    # haurien d ser iguals
    #print("img_path:\t", img_path)
    #print("patno:\t", patno)

    # agafam el cohort si el pacient tenia metadades al excel
    # segur q pandas té algo més ràpid x fer això
    if patno in labels_map:
        cohort = labels_map[patno]
        # x ara només interessen els PD i healthy
        if cohort in [1, 2]: #  cohort 1 són els PD i cohort 2 són els healthy
            label = 1 if cohort == 1 else 0 # els PD es marquen com 1 i els healthy es posen a 0
            #data_list.append({'path': img_path, 'label': label, 'cohort': cohort})
            data_list.append({'path': img_path, 'label': label})

# Create a final clean CSV for training
clean_df = pd.DataFrame(data_list)

# %%
# save the mapping to a file
clean_df.to_csv("ppmi_baseline_mapping.csv", index=False)
print("\nMapping saved to 'ppmi_baseline_mapping.csv'!")

# %% [markdown]
# Now we have a cleaned-up dataset with only PD or not.
# 
# # Train-test split
# 
# Now scikit learn comes in to train-split the dataset

# %%
from sklearn.model_selection import train_test_split

df = pd.read_csv("ppmi_baseline_mapping.csv")

# Separate the classes
pd_df = df[df['label'] == 1]
hc_df = df[df['label'] == 0]

# cut the PD since there are way more
pd_df_balanced = pd_df.sample(n=len(hc_df), random_state=42)

balanced_df = pd.concat([pd_df_balanced, hc_df])
print(f"Now the df contains {len(balanced_df)} patients, ({len(pd_df_balanced)} PD, and {len(hc_df)}) hc")

# new split
train_df, test_df = train_test_split(balanced_df, test_size=0.2, stratify=balanced_df['label'], random_state=42)

print(f"Original df: {len(df)} samples")
print(f"Balanced df: {len(balanced_df)} samples")
print(f"Training on: {len(train_df)} samples")
print(f"Testing on: {len(test_df)} samples")

# %% [markdown]
# # CNN
# 
# And once that's done, onto creating the CNN
# 

# %% [markdown]
# ## Arreglar `pytorch`/cuda:
# 
# Per algun motiu, cada pic em demana que em reinstali cuda amb la versió 12.6 tot i haver-ho fet. Desinstalar i reinstalar ho soluciona. Tarda ~1 min.
# ```sh
# pip3 uninstall torch torchvision
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# ```

# %%
#!pip3 uninstall torch torchvision -y
#!pip3 install torch torchvision
#!pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# %% [markdown]
# Pareix q tot i així no li omple xq falta una Kernel Image. La documentació no diu com solucionar.
# 
# 

# %%
import torch
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
print(f"PyTorch CUDA version: {torch.version.cuda}")

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class ParkinsonClassifier3D(nn.Module):
    def __init__(self):
        super(ParkinsonClassifier3D, self).__init__()
        # Input: 1 channel (scan), Output: 16 filters
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)

        # This part depends on image dimensions (91x109x91, isnt it?)
        # Global Average Pooling to avoid calculating flat dimensions
        self.gap = nn.AdaptiveAvgPool3d(1)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1) # Binary output (0 or 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = self.gap(x) # Flattens to [Batch, 64, 1, 1, 1]
        x = x.view(-1, 64) # Flattens to [Batch, 64]

        x = F.relu(self.fc1(x))
        # sigmoid for binary classification probability bcs why not
        x = torch.sigmoid(self.fc2(x))
        return x

model = ParkinsonClassifier3D().to("cuda") # Send to GPU

# %% [markdown]
# ## Loading the images
# 
# Nibabel is handy here since it takes the .nii.gz files and loads them to images.

# %%
import nibabel as nib
from torch.utils.data import Dataset, DataLoader

# %%
from monai.transforms import (
    Compose,
    LoadImaged,
    Resized,
    ScaleIntensityd,
    EnsureChannelFirstd,
    CenterSpatialCropd
)
from monai.data import Dataset as MonaiDataset

# Define the "Recipe" for your images
# We target a smaller size like 96x96x96 to save GPU memory
data_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    # Option 1: Resize (Squashes the image to fit 96x96x96)
    Resized(keys=["image"], spatial_size=(96, 96, 96)),
    # Option 2: CenterCrop (Trims the edges to fit 96x96x96)
    # seems to center around the commas but idk if this is adding noise
    # (different positionings, cropping varies, even scaling seems to be different too)
    #CenterSpatialCropd(keys=["image"], roi_size=(96, 96, 96)),
    ScaleIntensityd(keys=["image"]),
])

# Re-wrap your data
train_files = [{"image": p, "label": l} for p, l in zip(train_df['path'], train_df['label'])]
train_ds = MonaiDataset(data=train_files, transform=data_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

test_files = [{"image": p, "label": l} for p, l in zip(test_df['path'], test_df['label'])]
test_ds = MonaiDataset(data=test_files, transform=data_transforms)
test_loader = DataLoader(test_ds, batch_size=2, shuffle=True)

# %% [markdown]
# Once the images are all the same size, the datscandataset can be created (using nibabel to convert the nii.gz files into pytorch tensors just like before but now with the right image dimensions). INCORRECT

# Ok finaly: now we trainin

import os

# Folder to save models in case sth good comes out
os.makedirs("checkpoints", exist_ok=True)

import torch.optim as optim

# Initialize Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ParkinsonClassifier3D().to(device)

# Use BCEWithLogitsLoss because it is more numerically stable than putting a Sigmoid inside the model.
# NOTE: If you use this, REMOVE the 'torch.sigmoid' from your model's forward function!
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# We Trainin
num_epochs = 10

print(f"Starting training on {device}...") # no fos cosa q cuda falli i m surti cpu

best_val_loss = float('inf')

for epoch in range(num_epochs):
    # --- TRAINING PHASE ---
    model.train() 
    train_running_loss = 0.0
    
    for i, batch in enumerate(train_loader):
        images = batch["image"].to(device)
        labels = batch["label"].float().to(device).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()


        if (i + 1) % 50 == 0: # x veure com va
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_train_loss = train_running_loss / len(train_loader)

    # --- VALIDATION PHASE ---
    model.eval() 
    val_running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # No gradients needed, saves VRAM
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].float().to(device).view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            # Calculate Accuracy
            # Since we use BCEWithLogitsLoss, we need to apply sigmoid
            # or just check if output > 0 (logit 0 = probability 0.5)
            preds = (outputs > 0).float() 
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_running_loss / len(test_loader)
    val_acc = correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # --- SAVING THE BEST MODEL ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        print("--> Experimental 'best' model saved!")

    # Save a regular checkpoint every epoch
    torch.save(model.state_dict(), "checkpoints/latest_model.pth")
    print("-" * 30)


