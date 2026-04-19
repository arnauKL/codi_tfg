MODEL_NAME = "model_1"

import pandas as pd
import os
import glob

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

######## Get images and clean df

base_path = "/home/data/PPMI"
participants_file = os.path.join(base_path, "rawdata/participants.tsv")
curated_file = os.path.join(base_path, "documents/PPMI_Curated_Data_Cut_Public_20240729.xlsx")

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

# save the mapping to a file
clean_df.to_csv("ppmi_baseline_mapping.csv", index=False)
print("\nMapping saved to 'ppmi_baseline_mapping.csv'") # this is so I can reload 

df = pd.read_csv("ppmi_baseline_mapping.csv")

# Separate the classes
pd_df = df[df['label'] == 1]
hc_df = df[df['label'] == 0]

# make the df have the same ammount of PD as HC
pd_df_balanced = pd_df.sample(n=len(hc_df), random_state=42)

balanced_df = pd.concat([pd_df_balanced, hc_df])
print(f"Now the df contains {len(balanced_df)} patients, ({len(pd_df_balanced)} PD, and {len(hc_df)}) hc")
train_df, test_df = train_test_split(balanced_df, test_size=0.2, stratify=balanced_df['label'], random_state=42)

print(f"Original df: {len(df)} samples")
print(f"Balanced df: {len(balanced_df)} samples")
print(f"Training on: {len(train_df)} samples")
print(f"Testing on: {len(test_df)} samples")
 

#################################################### Process images
from torch.utils.data import DataLoader

from monai.data import Dataset as MonaiDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    CenterSpatialCropd,
    NormalizeIntensityd
)

# define transforms for the images
data_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    CenterSpatialCropd(keys=["image"], roi_size=(76, 76, 76)),
    NormalizeIntensityd(keys=["image"]),
])

# re-wrap data
train_files = [{"image": p, "label": l} for p, l in zip(train_df['path'], train_df['label'])]
train_ds = MonaiDataset(data=train_files, transform=data_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

test_files = [{"image": p, "label": l} for p, l in zip(test_df['path'], test_df['label'])]
test_ds = MonaiDataset(data=test_files, transform=data_transforms)
test_loader = DataLoader(test_ds, batch_size=2, shuffle=True)


#################################################### Define model

# cuda debug
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
print(f"PyTorch CUDA version: {torch.version.cuda}")

# architecture of the CNN
class ParkinsonClassifier3D(nn.Module):
    def __init__(self):
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
        self.dropout = nn.Dropout(0.3) # Drop 30% of neurons
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

# Initialize Model, Loss, and Optimizer

import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ParkinsonClassifier3D().to(device)

# Use BCEWithLogitsLoss because it is more numerically stable than putting a Sigmoid inside the model.
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

####### save stuff into a csv so I can see later

import csv
log_file = "logs/training_log_" + MODEL_NAME + ".csv"

# Write header
with open(log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])

#################################################### We Trainin
num_epochs = 80

print(f"Starting training on {device}") # no fos cosa q cuda falli i m surti cpu

best_val_loss = float('inf')

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs} {100*(epoch+1)/num_epochs}% done", end="")
    
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

    avg_train_loss = train_running_loss / len(train_loader)

    # validació
    model.eval() 
    val_running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # no gradients needed, saves VRAM
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].float().to(device).view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            # find accuracy
            # oju sigmoid, we're using logits
            # (logit 0 = probability 0.5)
            preds = (outputs > 0).float() 
            correct += (preds == labels).sum().item()
            total += labels.size(0) # cheap

    avg_val_loss = val_running_loss / len(test_loader)
    val_acc = correct / total

    # save numbers to csv
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, avg_train_loss, avg_val_loss, val_acc])

    print(f" [Train Loss: {avg_train_loss:.4f}; Val Loss: {avg_val_loss:.4f}; Val Acc: {val_acc:.4f}]", end="")

    # keep the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "checkpoints/best_" + MODEL_NAME + ".pth")
        print("\tNew best model saved", end="")


################################################### avaluació
model.eval()

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import numpy as np

all_preds = []
all_labels = []

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device).view(-1, 1)
        
        outputs = model(images)
        
        # Get probabilities (0 to 1)
        probs = torch.sigmoid(outputs) # If you removed sigmoid from the model class
        
        # Get hard predictions (0 or 1)
        preds = (outputs > 0).float() # to float bcs pytorch is picky and floats are floats

        #print(f"image {i}\nprobs:\t{probs}\npreds:\t{preds}")
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert to clean numpy arrays
all_preds = np.array(all_preds).flatten()
all_labels = np.array(all_labels).flatten()

tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

print("\n")
print("=" * 30)
print("Results of '" + MODEL_NAME + "':")
print(f"TP: {tp}\tPD detected")
print(f"TN: {tn}\tHealthy detected")
print(f"FP: {fp}\tHealthy missdiagnosed")
print(f"FN: {fn}\tPD missed")

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print("\nFull Report:")
print(classification_report(all_labels, all_preds, target_names=['Healthy', 'PD']))
