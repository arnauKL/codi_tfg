# evaluate/bootstrap_eval.py
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.utils import resample

def bootstrap_evaluate(model, loader, device, n_bootstrap=100, seed=42):
    """Get all predictions first, then bootstrap over them."""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()
            outputs = torch.sigmoid(model(images)).cpu().numpy().flatten()
            all_probs.extend(outputs)
            all_labels.extend(labels)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    rng = np.random.RandomState(seed)
    metrics = []
    for _ in range(n_bootstrap):
        idx = resample(range(len(all_labels)), random_state=rng.randint(0, 9999))
        probs_b = all_probs[idx]
        labels_b = all_labels[idx]
        preds_b = (probs_b > 0.5).astype(int)
        metrics.append({
            "f1":  f1_score(labels_b, preds_b, zero_division=0),
            "auc": roc_auc_score(labels_b, probs_b),
            "acc": accuracy_score(labels_b, preds_b),
        })

    return pd.DataFrame(metrics)

# evaluate/plot_comparison.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_metric_boxplots(results_dict, metric="f1", output_path=None):
    """
    results_dict: {"ModelName": pd.DataFrame with metric columns, ...}
    """
    fig, ax = plt.subplots(figsize=(max(6, len(results_dict) * 1.5), 5))

    data = [df[metric].values for df in results_dict.values()]
    labels = list(results_dict.keys())

    bp = ax.boxplot(data, patch_artist=True, labels=labels)

    colors = plt.cm.Set2.colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel(metric.upper())
    ax.set_title(f"Model comparison — {metric.upper()}")
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()