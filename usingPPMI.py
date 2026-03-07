#!/usr/bin/env python

import datetime
import logging
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import *
from torch.utils.data import DataLoader, Dataset, Subset, random_split

# Hiperparàmetres

VAL_RATIO = 0.2
TEST_RATIO = 0.2
TRAIN_RATIO = 1 - VAL_RATIO - TEST_RATIO
DATA_PARALELL = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = "HCP"
PHASE = "TRAIN"
K_FOLDS = 5

TASK = "classification"
ROOT_DIR = "path/to/dataset/HCP"
LABEL_DIR = "path/to/metadata/metadata.csv"
LOG_DIR = "path/to/scripts/logs"
timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
RESULT_DIR = f"path/to/scripts/results/run_{timestamp}"
os.makedirs(RESULT_DIR, exist_ok=True)
TEST_BATCH_SIZE = 2

BATCH_SIZE = 16
LEARNING_RATE = 0.0001
NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 5

NP_SEED = 42
TORCH_SEED = 36

# inicialitzar les seeds
np.random.seed(NP_SEED)
torch.manual_seed(TORCH_SEED)


# Configurar el logger (encara no sé molt bé perquè és)
def setup_logger(logs_dir=LOG_DIR, dataset=None, phase=None):
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger("RunLogger")
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        file_handler = logging.FileHandler(
            os.path.join(logs_dir, f"{phase}_{dataset}_{timestamp}.log")
        )
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


LOGGER = setup_logger(logs_dir=LOG_DIR, phase=PHASE, dataset=DATASET)


# Dataset class.
class Fieldmapdata(Dataset):
    """
    Dataset class for loading neuroimaging data and associated labels for classification or regression tasks.

    Attributes:
        root_dir (str): Path to the directory containing image data files (.nii or .nii.gz).
        label_dir (str): Path to the CSV file containing labels for each subject.
        task (str): Specifies the task type: classification (Gender) or regression (Age).
    """

    def __init__(self, root_dir, label_dir, task="classification"):
        self.labels_df = self.load_labels(label_dir)  # load the labels
        self.samples = self.make_dataset(
            root_dir, task=task
        )  # assign the labels to each file in the root_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        nifti_data = nib.load(img_path)
        data = nifti_data.get_fdata()
        image_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(
            0
        )  # unsqueeze to add channel dimension
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        return image_tensor, label_tensor

    def make_dataset(self, root_dir, task=None):
        samples = []
        labels_df = self.labels_df
        for root, _, fnames in os.walk(root_dir):
            for fname in fnames:
                # no cal això, sempre serà .nii.gz
                if fname.endswith(".nii.gz") or fname.endswith(".nii"):
                    path = os.path.join(root, fname)
                    id_ = self.extract_id_from_filename(fname)
                    try:
                        if task == "regression":
                            label = labels_df[labels_df["Subject"] == id_]["Age"].iloc[
                                0
                            ]
                        if task == "classification":
                            label = labels_df[labels_df["Subject"] == id_][
                                "Gender"
                            ].iloc[0]
                        samples.append((path, label))
                    except:
                        continue
        return samples

    def extract_id_from_filename(self, fname):
        """match the filename to key to query in the labels dictionary"""
        fname = fname.replace("sub-", "")
        if fname.endswith("_ad.nii.gz"):
            id_ = fname.replace("_ad.nii.gz", "")
        elif fname.endswith("_rd.nii.gz"):
            id_ = fname.replace("_rd.nii.gz", "")
        elif fname.endswith("_adc.nii.gz"):
            id_ = fname.replace("_adc.nii.gz", "")
        elif fname.endswith("_fa.nii.gz"):
            id_ = fname.replace("_fa.nii.gz", "")
        elif fname.endswith(".nii.gz"):
            id_ = fname.replace(".nii.gz", "")
        return id_

    def load_labels(self, label_path):
        """tip: use astype(required datatype)"""
        df = pd.read_csv(label_path)
        df_filtered = df[["Subject", "Gender", "Age"]].copy()
        df_filtered["Gender"] = df_filtered["Gender"].map({"M": 0, "F": 1}).astype(int)
        df_filtered["Age"] = (
            df_filtered["Age"]
            .apply(
                lambda x: (
                    (int(x.split("-")[0]) + int(x.split("-")[1])) // 2
                    if "-" in x
                    else int(x[:-1])
                )
            )
            .astype(float)
        )
        # In HCP-Y age is a bin of 4 years, here i can assigning the average value of the bin range to each subject
        df_filtered["Subject"] = df_filtered["Subject"].astype(str)
        return df_filtered
