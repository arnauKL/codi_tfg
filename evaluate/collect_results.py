# evaluate/collect_results.py
import os, json, re
import pandas as pd

OUTPUTS_DIR = "outputs"
rows = []

for run in os.listdir(OUTPUTS_DIR):
    run_path = os.path.join(OUTPUTS_DIR, run)
    config_path = os.path.join(run_path, "config.json")
    results_path = os.path.join(run_path, "final_results.txt")

    if not os.path.exists(config_path) or not os.path.exists(results_path):
        continue

    with open(config_path) as f:
        config = json.load(f)

    with open(results_path) as f:
        text = f.read()

    # Parse classification_report text
    # Grab the weighted avg line
    match = re.search(
        r"weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text
    )
    if not match:
        continue

    rows.append({
        "run": run,
        "model_name": config.get("model_name", run),
        "precision": float(match.group(1)),
        "recall":    float(match.group(2)),
        "f1":        float(match.group(3)),
        # add whatever config params you want to track:
        "roi_size":  str(config.get("roi_size")),
        "balanced":  "Unbalanced" not in run,
    })

df = pd.DataFrame(rows)
df.to_csv("evaluate/all_results.csv", index=False)
print(df[["model_name", "f1", "precision", "recall"]].sort_values("f1", ascending=False))