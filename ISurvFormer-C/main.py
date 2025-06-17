import os
import json
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import optuna
from optuna.trial import TrialState

from data_load import load_data
from hyper_search import objective, train_model_cv
from model import DynamicSurvTransformer, generate_mask
from utils import compute_ibs_from_risk

warnings.filterwarnings("ignore")

# -------------
# Configuration 
# -------------
import os

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Read from environment variables (with defaults)
DATA_NAME = os.environ.get("ISURV_DATA", "AIDS")
TRIALS = int(os.environ.get("ISURV_TRIALS", 30))
MAX_EPOCHS = int(os.environ.get("ISURV_EPOCHS", 200))
PATIENCE = int(os.environ.get("ISURV_PATIENCE", 50))
DEVICE = os.environ.get("ISURV_DEVICE", "cuda")
SAVE_DIR = os.environ.get("ISURV_SAVE_DIR", "ISurvFormer-C/result")

# Paths
DATA_PATH = f"data/{DATA_NAME.lower()}.csv"
RESULT_ROOT = os.path.join(SAVE_DIR, DATA_NAME)
PARAM_PATH = os.path.join(RESULT_ROOT, "params.json")
CV_RESULT_PATH = os.path.join(RESULT_ROOT, "final_cv_result.txt")
MODEL_SAVE_DIR = os.path.join(RESULT_ROOT, "saved_models")
CLUSTER_CSV_PATH = os.path.join(RESULT_ROOT, "cluster_assignments.csv")
IBS_RESULT_PATH = os.path.join(RESULT_ROOT, "ibs_result.txt")


# ------------------------------------------------------------------
# Step 1 ─ Load data
# ------------------------------------------------------------------
print("📦 Loading data …")
x_all, time_all, lengths_all, duration_all, event_all, D, _ = load_data(DATA_PATH)

# Build masks
x_true = x_all.clone()                        # backup for imputation loss
mask_raw = ~torch.isnan(x_all)                # 1 = observed, 0 = missing
x_all = x_all.clone()
x_all[~mask_raw] = 0.0                        # fill NaNs with 0 for input
mask_pad = generate_mask(lengths_all, x_all.size(1))  # [B, T]

# Train/val split (only for IBS later)
idx_all = np.arange(x_all.size(0))
train_idx, val_idx = train_test_split(idx_all, test_size=0.2, random_state=SEED)
print(f"✅ Train samples: {len(train_idx)} | Val samples: {len(val_idx)}")

# ------------------------------------------------------------------
# Step 2 ─ Hyper-parameter search (Optuna) or load cached parameters
# ------------------------------------------------------------------
if os.path.exists(PARAM_PATH):
    print(f"📂 Loading hyper-parameters from {PARAM_PATH}")
    with open(PARAM_PATH, "r") as f:
        best_params = json.load(f)
else:
    print("🔍 Running Optuna search …")

    def quick_objective(trial):
        return objective(
            trial,
            D,
            x_all=x_all,
            x_true_all=x_true,
            mask_raw_all=mask_raw,
            time_all=time_all,
            mask_pad_all=mask_pad,
            lengths_all=lengths_all,
            duration_all=duration_all,
            event_all=event_all,
        )

    study = optuna.create_study(direction="minimize")
    max_trials = 30
    while len(study.trials) < max_trials:
        trial_id = len(study.trials)
        study.optimize(quick_objective, n_trials=1)
        last_state = study.trials[-1].state
        msg = "COMPLETED" if last_state is TrialState.COMPLETE else last_state.name
        print(f"Trial {trial_id} → {msg}")

    best_params = study.best_trial.params
    os.makedirs(os.path.dirname(PARAM_PATH), exist_ok=True)
    with open(PARAM_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"✅ Best params saved to {PARAM_PATH}")

# ------------------------------------------------------------------
# Step 3 ─ Cross-validation training
# ------------------------------------------------------------------
def model_fn():
    return DynamicSurvTransformer(
        input_dim=D,
        d_model=best_params["d_model"],
        nhead=best_params["nhead"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
        mlp_hidden_ratio=best_params["mlp_hidden_ratio"],
        mlp_num_layers=best_params["mlp_num_layers"],
        activation=best_params["activation"],
        risk_hidden_dim=best_params.get("risk_hidden_dim", 64),
        risk_num_layers=best_params.get("risk_num_layers", 2),
        risk_activation=best_params.get("risk_activation", "ReLU"),
        num_clusters=best_params.get("num_clusters", 2),
    )

def optimizer_fn(model):
    return torch.optim.Adam(model.parameters(), lr=best_params["lr"])

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
mean_cidx = train_model_cv(
    model_fn=model_fn,
    optimizer_fn=optimizer_fn,
    alpha=best_params.get("alpha", 1.0),
    beta=best_params.get("beta", 1.0),
    gamma=best_params.get("gamma", 1.0),
    max_epochs=200,
    patience=50,
    x_all=x_all,
    x_true_all=x_true,
    mask_raw_all=mask_raw,
    time_all=time_all,
    mask_pad_all=mask_pad,
    lengths_all=lengths_all,
    duration_all=duration_all,
    event_all=event_all,
    n_splits=10,
    save_path=CV_RESULT_PATH,
    model_save_dir=MODEL_SAVE_DIR,
    vis=True,
)
print(f"\n🎯 Final 10-fold C-index: {mean_cidx:.4f} → saved to {CV_RESULT_PATH}")

# ------------------------------------------------------------------
# Step 4 ─ Predict and save cluster assignments
# ------------------------------------------------------------------
print("📌 Predicting cluster assignments (model-average) …")
num_folds = 10

# Average parameters across folds
avg_state = OrderedDict()
for fold in range(1, num_folds + 1):
    path = os.path.join(MODEL_SAVE_DIR, f"fold{fold}_model.pt")
    if not os.path.exists(path):
        continue
    sd = torch.load(path, map_location="cpu")
    for k, v in sd.items():
        avg_state[k] = avg_state.get(k, 0) + v
for k in avg_state:
    avg_state[k] /= num_folds

model = model_fn()
model.load_state_dict(avg_state)
model.eval()

with torch.no_grad():
    _, _, _, _, q_probs = model(x_all, mask_raw, time_all, mask_pad)
    probs = q_probs.cpu().numpy()
    assignments = probs.argmax(axis=1)

df_cluster = pd.DataFrame({"sample_index": np.arange(len(assignments)),
                           "cluster": assignments})
for k in range(probs.shape[1]):
    df_cluster[f"prob_c{k}"] = probs[:, k]

os.makedirs(os.path.dirname(CLUSTER_CSV_PATH), exist_ok=True)
df_cluster.to_csv(CLUSTER_CSV_PATH, index=False)
print(f"✅ Cluster CSV saved to {CLUSTER_CSV_PATH}")

# ------------------------------------------------------------------
# Step 5 ─ Integrated Brier Score (IBS) evaluation
# ------------------------------------------------------------------
print("📊 Calculating IBS …")
ibs_vals = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold in range(1, 11):
    model_path = os.path.join(MODEL_SAVE_DIR, f"fold{fold}_model.pt")
    if not os.path.exists(model_path):
        print(f"⚠️ Missing {model_path} → skipped")
        continue

    model = model_fn()
    model.load_state_dict(torch.load(model_path))
    model.to(device).eval()

    x_val_t = x_all[val_idx].to(device)
    t_val_t = time_all[val_idx].to(device)
    mask_raw_t = mask_raw[val_idx].to(device)
    mask_pad_t = mask_pad[val_idx].to(device)
    dur_t = duration_all[val_idx].to(device)
    evt_t = event_all[val_idx].to(device)
    len_t = lengths_all[val_idx].to(device)

    ibs = compute_ibs_from_risk(model, x_val_t, mask_raw_t, t_val_t, mask_pad_t,
                                len_t, dur_t, evt_t)
    if not np.isnan(ibs):
        ibs_vals.append(ibs)
        print(f"Fold {fold}: IBS = {ibs:.4f}")

if ibs_vals:
    os.makedirs(os.path.dirname(IBS_RESULT_PATH), exist_ok=True)
    with open(IBS_RESULT_PATH, "w") as f:
        for i, v in enumerate(ibs_vals, 1):
            f.write(f"Fold {i}: IBS = {v:.4f}\n")
        f.write(f"\nMean IBS = {np.mean(ibs_vals):.4f}\n")
        f.write(f"Std  IBS = {np.std(ibs_vals):.4f}\n")
    print(f"✅ IBS results saved to {IBS_RESULT_PATH}")
else:
    print("⚠️ No IBS values computed.")
