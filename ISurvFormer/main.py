import os
import json
import warnings

import numpy as np
import torch
import optuna
from sklearn.model_selection import train_test_split

from dataset import load_data
from model import DynamicSurvTransformer, generate_mask
from tuning import objective_tune_no_cv
from utils import train_model_cv

warnings.filterwarnings("ignore")


def _ensure_dir(path: str) -> None:
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def run_experiment(
    data: str,
    data_root: str = "./data",
    res_root: str = "./result",
    seed: int = 42,
    trials: int = 10,
    max_epochs_tune: int = 80,
    patience_tune: int = 20,
    max_epochs_cv: int = 100,
    patience_cv: int = 10,
) -> None:
    """
    Run a full pipeline for a single dataset:
        1) Hyperparameter tuning on a train/validation split (no leakage).
        2) Final 10-fold CV using the best hyperparameters.

    Args:
        data:             Dataset name (recommended uppercase, e.g. "LINEAR_SHORT_TERM").
        data_root:        Root directory for input CSV files.
        res_root:         Root directory for saving results.
        seed:             Random seed for reproducibility.
        trials:           Number of Optuna trials.
        max_epochs_tune:  Max epochs during tuning.
        patience_tune:    Early stopping patience during tuning.
        max_epochs_cv:    Max epochs during final 10-fold CV.
        patience_cv:      Early stopping patience during final 10-fold CV.
    """
    # Fix random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Normalize dataset name: directory uses upper case, file name uses lower case
    data_name = data.upper()
    data_name_for_path = data_name.lower()

    data_path = os.path.join(data_root, f"{data_name_for_path}.csv")
    param_path = os.path.join(res_root, data_name, "params.json")

    _ensure_dir(os.path.dirname(param_path))

    print("\n" + "=" * 90)
    print(f"DATASET: {data_name}")
    print(f"Loading data from: {data_path}")

    # load_data should return torch tensors
    (
        x_all,
        time_all,
        lengths_all,
        duration_all,
        event_all,
        D,
        mask_pad,
        mask_raw,
    ) = load_data(data_path)

    # Build "true" and "input" versions (no NaNs go into the model)
    x_true = x_all.clone()              # contains NaNs (ground truth for imputation)
    mask_raw = ~torch.isnan(x_all)      # True where observed
    x_input = x_all.clone()
    x_input[~mask_raw] = 0.0            # fill missing with 0 for model input

    # Padding mask from lengths
    mask_pad = generate_mask(lengths_all, x_all.size(1))  # [B, T] bool

    # Train/validation split for tuning only (no leakage into CV)
    idx_all = np.arange(x_input.shape[0])
    train_idx, tune_val_idx = train_test_split(
        idx_all,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )

    print(f"Tuning split | Train: {len(train_idx)} | Tune-Val: {len(tune_val_idx)}")

    # Prepare tuning data (no leakage)
    x_tr, x_va = x_input[train_idx], x_input[tune_val_idx]
    x_true_tr, x_true_va = x_true[train_idx], x_true[tune_val_idx]
    mask_raw_tr, mask_raw_va = mask_raw[train_idx], mask_raw[tune_val_idx]
    time_tr, time_va = time_all[train_idx], time_all[tune_val_idx]
    mask_pad_tr, mask_pad_va = mask_pad[train_idx], mask_pad[tune_val_idx]
    lengths_tr, lengths_va = lengths_all[train_idx], lengths_all[tune_val_idx]
    duration_tr, duration_va = duration_all[train_idx], duration_all[tune_val_idx]
    event_tr, event_va = event_all[train_idx], event_all[tune_val_idx]

    # ------------------------------------------------------------------
    # Step 1: Hyperparameter search (no CV, no leakage from CV folds)
    # ------------------------------------------------------------------
    if os.path.exists(param_path):
        print(f"Found saved hyperparameters: {param_path} (loading)")
        with open(param_path, "r", encoding="utf-8") as f:
            best_params = json.load(f)
    else:
        print("No saved hyperparameters found, running Optuna tuning (no CV)...")

        def quick_objective(trial):
            return objective_tune_no_cv(
                trial=trial,
                D=D,
                x_train=x_tr,
                x_true_train=x_true_tr,
                mask_raw_train=mask_raw_tr,
                time_train=time_tr,
                mask_pad_train=mask_pad_tr,
                lengths_train=lengths_tr,
                duration_train=duration_tr,
                event_train=event_tr,
                x_val=x_va,
                x_true_val=x_true_va,
                mask_raw_val=mask_raw_va,
                time_val=time_va,
                mask_pad_val=mask_pad_va,
                lengths_val=lengths_va,
                duration_val=duration_va,
                event_val=event_va,
                max_epochs=max_epochs_tune,
                patience=patience_tune,
                seed=seed,
            )

        study = optuna.create_study(direction="minimize")
        study.optimize(
            quick_objective,
            n_trials=trials,
            gc_after_trial=True,
            catch=(ValueError, FloatingPointError, RuntimeError),
        )

        best_params = study.best_trial.params
        with open(param_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)

        print(f"Best hyperparameters saved to {param_path}")

    # ------------------------------------------------------------------
    # Step 2: Final 10-fold CV with best hyperparameters (no leakage)
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
        )

    def optimizer_fn(model):
        return torch.optim.Adam(model.parameters(), lr=best_params["lr"])

    cv_result_path = os.path.join(res_root, data_name, "final_cv_result.txt")
    ibs_result_path = os.path.join(res_root, data_name, "ibs_cv_result.txt")
    time_result_path = os.path.join(res_root, data_name, "time_to_best_cv_result.txt")
    split_save_path = os.path.join(res_root, data_name, "cv_splits.json")
    model_save_dir = os.path.join(res_root, "saved_models", data_name)

    _ensure_dir(os.path.dirname(cv_result_path))
    _ensure_dir(model_save_dir)

    print("\nRunning final 10-fold cross-validation (C-index + IPCW-IBS)...")
    print("|-------------------------------------------------------------------------|")

    mean_cidx, mean_ibs = train_model_cv(
        model_fn=model_fn,
        optimizer_fn=optimizer_fn,
        alpha=best_params.get("alpha", 1.0),
        beta=best_params.get("beta", 1.0),
        max_epochs=max_epochs_cv,
        patience=patience_cv,
        # Important: use x_input (no NaNs) for model input
        x_all=x_input,
        x_true_all=x_true,
        mask_raw_all=mask_raw,
        time_all=time_all,
        mask_pad_all=mask_pad,
        lengths_all=lengths_all,
        duration_all=duration_all,
        event_all=event_all,
        n_splits=10,
        save_path=cv_result_path,
        model_save_dir=model_save_dir,
        split_save_path=split_save_path,
        ibs_save_path=ibs_result_path,
        time_save_path=time_result_path,
        vis=True,
    )

    print(f"\nFinal 10-fold CV mean C-index: {mean_cidx:.4f}")
    print(f"Final 10-fold CV mean IPCW-IBS: {mean_ibs:.4f}")
    print(f"C-index per-fold saved to: {cv_result_path}")
    print(f"IPCW-IBS per-fold saved to: {ibs_result_path}")
    print(f"CV splits saved to: {split_save_path}")
