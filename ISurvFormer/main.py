import os
import json
import torch
import numpy as np
import optuna
import warnings
from optuna.trial import TrialState
from sklearn.model_selection import train_test_split

from data_load import load_data
from hyper_search import objective, train_model_cv
from model import DynamicSurvTransformer, generate_mask
from utils import compute_ibs

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Read from run.py environment settings if available
DATA_NAME = os.getenv("ISURV_DATA", "AIDS")
MAX_TRIALS = int(os.getenv("ISURV_TRIALS", 30))
MAX_EPOCH = int(os.getenv("ISURV_EPOCHS", 200))
PATIENCE = int(os.getenv("ISURV_PATIENCE", 50))
DEVICE = torch.device(os.getenv("ISURV_DEVICE", "cuda") if torch.cuda.is_available() else "cpu")
SAVE_ROOT = os.getenv("ISURV_SAVE_DIR", "ISurvFormer/result")

# Construct derived paths
DATA_NAME_FOR_PATH = DATA_NAME.lower()
DATA_PATH = f"data/{DATA_NAME_FOR_PATH}.csv"
PARAM_PATH = os.path.join(SAVE_ROOT, DATA_NAME, "params.json")
CV_RESULT_PATH = os.path.join(SAVE_ROOT, DATA_NAME, "cindex_result.txt")
IBS_RESULT_PATH = os.path.join(SAVE_ROOT, DATA_NAME, "ibs_result.txt")
MODEL_SAVE_DIR = os.path.join(SAVE_ROOT, "saved_models", DATA_NAME)

os.makedirs(os.path.dirname(PARAM_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CV_RESULT_PATH), exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# === Step 1: Data Loading & Preprocessing ===
print("📦 Loading data...")
x_all, time_all, lengths_all, duration_all, event_all, D, mask_pad, mask_raw_all = load_data(DATA_PATH)

# Backup true input
x_true = x_all.clone()
mask_raw = mask_raw_all
x_all = x_all.clone()
x_all[~mask_raw] = 0.0

# Time padding mask
mask_pad = generate_mask(lengths_all, x_all.size(1))  # [B, T]

# Train/Val Split (for sanity checks)
index = np.arange(x_all.shape[0])
train_idx, val_idx = train_test_split(index, test_size=0.2, random_state=SEED)

x_train, x_val = x_all[train_idx], x_all[val_idx]
x_true_train, x_true_val = x_true[train_idx], x_true[val_idx]
mask_raw_train, mask_raw_val = mask_raw[train_idx], mask_raw[val_idx]
time_train, time_val = time_all[train_idx], time_all[val_idx]
mask_pad_train, mask_pad_val = mask_pad[train_idx], mask_pad[val_idx]
lengths_train, lengths_val = lengths_all[train_idx], lengths_all[val_idx]
duration_train, duration_val = duration_all[train_idx], duration_all[val_idx]
event_train, event_val = event_all[train_idx], event_all[val_idx]

print(f"✅ Train samples: {len(train_idx)} | Val samples: {len(val_idx)}")

# === Step 2: Hyperparameter Search (Optuna) ===
if os.path.exists(PARAM_PATH):
    print(f"📂 Found saved hyperparameters in {PARAM_PATH}, loading...")
    with open(PARAM_PATH, "r") as f:
        best_params = json.load(f)
else:
    print("🔍 Running Optuna hyperparameter search...")

    def quick_objective(trial):
        return objective(trial, D,
                         x_all=x_all, x_true_all=x_true,
                         mask_raw_all=mask_raw, time_all=time_all,
                         mask_pad_all=mask_pad, lengths_all=lengths_all,
                         duration_all=duration_all, event_all=event_all)

    study = optuna.create_study(direction="minimize")

    MAX_TRIALS = 30
    actual_trials = 0
    trial_id = 0

    while actual_trials < MAX_TRIALS:
        study.optimize(quick_objective, n_trials=1)
        last_trial = study.trials[-1]

        if last_trial.state == TrialState.COMPLETE:
            actual_trials += 1
            print(f"✅ Trial {trial_id} completed ({actual_trials}/{MAX_TRIALS})")
        elif last_trial.state == TrialState.PRUNED:
            print(f"⚠️ Trial {trial_id} pruned.")
        else:
            print(f"❓ Trial {trial_id} ended with state: {last_trial.state.name}")

        trial_id += 1

    best_params = study.best_trial.params
    with open(PARAM_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"✅ Best hyperparameters saved to {PARAM_PATH}")

# === Step 3: Cross-Validation Training ===
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
        risk_activation=best_params.get("risk_activation", "ReLU")
    )

def optimizer_fn(model):
    return torch.optim.Adam(model.parameters(), lr=best_params["lr"])

print("🚀 Running 10-fold Cross-Validation...")
mean_cidx = train_model_cv(
    model_fn=model_fn,
    optimizer_fn=optimizer_fn,
    alpha=best_params.get("alpha", 1.0),
    beta=best_params.get("beta", 1.0),
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
    vis=True
)
print(f"\n🎯 Final 10-fold CV C-index: {mean_cidx:.4f}")
print(f"📄 Results saved to {CV_RESULT_PATH}")

# === Step 4: Evaluate Integrated Brier Score (IBS) ===
print("\n📊 Calculating Integrated Brier Score (IBS)...")
ibs_list = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold in range(1, 11):
    model_path = os.path.join(MODEL_SAVE_DIR, f"fold{fold}_model.pt")
    if not os.path.exists(model_path):
        print(f"⚠️ Missing model: {model_path}, skipping...")
        continue

    model = torch.load(model_path, map_location=device, weights_only=False).to(device)
    ibs = compute_ibs(model, x_all, mask_raw, time_all, mask_pad, lengths_all, duration_all, event_all)
    ibs_list.append(ibs)
    print(f"✅ Fold {fold}: IBS = {ibs:.4f}")

# Save IBS results
with open(IBS_RESULT_PATH, "w") as f:
    for i, val in enumerate(ibs_list, 1):
        f.write(f"Fold {i}: IBS = {val:.4f}\n")
    f.write(f"\nMean IBS = {np.mean(ibs_list):.4f}\n")
    f.write(f"Std  IBS = {np.std(ibs_list):.4f}\n")

print(f"📄 IBS results saved to {IBS_RESULT_PATH}")
