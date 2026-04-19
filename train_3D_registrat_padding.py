#intento d fer el codi una mica més modular
import os
import json
import datetime
import csv
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.data import Dataset as MonaiDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# import modules
# ==============
from src.architectures import ParkinsonClassifier2D
from src.transforms import get_2d_sum_striatum_transforms

CONFIG = {
    "model_name": "2D_nomes_striatum_rawderiv",
    "data_path": "ppmi_raw_n_derivative_mapping.csv",
    "roi_size": (76, 76, 76),
    "batch_size": 2,
    "lr": 0.0001,
    "epochs": 100,
    "dropout": 0.3,
    "val_size": 0.2,
    "random_seed": 42
}

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"{CONFIG['model_name']}_{timestamp}"
output_dir = os.path.join("outputs", run_name)
datadir = os.path.join("data", CONFIG['data_path'])
os.makedirs(output_dir, exist_ok=True)

# Save the config to a file
with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump(CONFIG, f, indent=4)

print(f"Starting run: {run_name}")
print(f"Results will be saved to: {output_dir}")


# preparing data
# ==============
df = pd.read_csv(datadir)

# Balance the dataset (PD vs HC)
pd_df = df[df['label'] == 1]
hc_df = df[df['label'] == 0]
pd_df_balanced = pd_df.sample(n=len(hc_df), random_state=CONFIG["random_seed"])
balanced_df = pd.concat([pd_df_balanced, hc_df])
print(f'balanced df has {len(balanced_df)} images')

train_df, test_df = train_test_split(
    balanced_df, 
    test_size=CONFIG["val_size"], 
    stratify=balanced_df['label'], 
    random_state=CONFIG["random_seed"]
)

# Loaders
train_files = [{"image": p, "label": l} for p, l in zip(train_df['path'], train_df['label'])]
train_ds = MonaiDataset(data=train_files, transform=get_2d_sum_striatum_transforms(CONFIG["roi_size"]))
train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)

test_files = [{"image": p, "label": l} for p, l in zip(test_df['path'], test_df['label'])]
test_ds = MonaiDataset(data=test_files, transform=get_2d_sum_striatum_transforms(CONFIG["roi_size"]))
test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"])


# Model, loss and optimizer
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ParkinsonClassifier2D(dropout_rate=CONFIG["dropout"]).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])


# Training
# ========

log_file = os.path.join(output_dir, "training_log.csv")
best_val_loss = float('inf')

with open(log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])

print(f"Starting training on {device}")

for epoch in range(CONFIG["epochs"]):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        images, labels = batch["image"].to(device), batch["label"].float().to(device).view(-1, 1)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch["image"].to(device), batch["label"].float().to(device).view(-1, 1)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            preds = (outputs > 0).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(test_loader)
    val_acc = correct / total

    # Save to log
    with open(log_file, mode='a', newline='') as f:
        csv.writer(f).writerow([epoch+1, avg_train_loss, avg_val_loss, val_acc])

    print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}\tVal Acc: {val_acc:.4f}", end='')

    # Save Best Model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
        print("\tNew best model", end='')


# Evaluation
# ==========
model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth")))
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        images, labels = batch["image"].to(device), batch["label"].to(device).view(-1, 1)
        outputs = model(images)
        all_preds.extend((outputs > 0).float().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

report = classification_report(all_labels, all_preds, target_names=['Healthy', 'PD'])

with open(os.path.join(output_dir, "final_results.txt"), "w") as f:
    f.write(report)

print("\nTraining done")