import optuna
import torch
import numpy as np

from model import DynamicSurvTransformer
from utils import train_one_and_eval_cindex


def objective_tune_no_cv(
    trial, D, x_train,
    x_true_train, mask_raw_train, time_train,
    mask_pad_train, lengths_train, duration_train,
    event_train, x_val, x_true_val,
    mask_raw_val, time_val, mask_pad_val,
    lengths_val, duration_val, event_val,
    max_epochs: int = 80, patience: int = 20, seed: int = 42,
):
    TrialPruned = optuna.exceptions.TrialPruned

    # ---- search space ----
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    nhead = trial.suggest_int("nhead", 1, 8)
    if d_model % nhead != 0:
        raise TrialPruned()

    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    mlp_hidden_ratio = trial.suggest_categorical("mlp_hidden_ratio", [0.5, 1.0, 2.0])
    mlp_num_layers = trial.suggest_int("mlp_num_layers", 1, 4)
    activation = trial.suggest_categorical(
        "activation",
        ["ReLU", "GELU", "SELU", "LeakyReLU"],
    )

    risk_hidden_dim = trial.suggest_categorical(
        "risk_hidden_dim",
        [64, 128, 256],
    )
    risk_num_layers = trial.suggest_int("risk_num_layers", 2, 4)
    risk_activation = trial.suggest_categorical(
        "risk_activation",
        ["ReLU", "LeakyReLU", "SELU", "GELU"],
    )

    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    beta = trial.suggest_float("beta", 0.1, 5.0, log=True)

    # ---- build model/optimizer ----
    model = DynamicSurvTransformer(
        input_dim=D,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        mlp_hidden_ratio=mlp_hidden_ratio,
        mlp_num_layers=mlp_num_layers,
        activation=activation,
        risk_hidden_dim=risk_hidden_dim,
        risk_num_layers=risk_num_layers,
        risk_activation=risk_activation,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- train on train split, early stop by val C-index ----
    try:
        best_cidx, _best_state = train_one_and_eval_cindex(
            model=model,
            optimizer=optimizer,
            x_train=x_train,
            mask_raw_train=mask_raw_train,
            x_true_train=x_true_train,
            t_train=time_train,
            mask_pad_train=mask_pad_train,
            durations_train=duration_train,
            events_train=event_train,
            lengths_train=lengths_train,
            x_val=x_val,
            mask_raw_val=mask_raw_val,
            x_true_val=x_true_val,
            t_val=time_val,
            mask_pad_val=mask_pad_val,
            durations_val=duration_val,
            events_val=event_val,
            lengths_val=lengths_val,
            alpha=alpha,
            beta=beta,
            max_epochs=max_epochs,
            patience=patience,
            vis=False,
        )
    except (ValueError, FloatingPointError, RuntimeError, ZeroDivisionError):
        # Numerical instability / invalid lifelines input / training divergence -> prune trial.
        raise TrialPruned()
    except Exception:
        # Fallback: any unexpected error also prunes the trial to avoid breaking the search.
        raise TrialPruned()

    # Prune invalid or degenerate C-index values.
    if (best_cidx is None) or (not np.isfinite(best_cidx)) or (best_cidx <= 0.0):
        raise TrialPruned()

    # direction = minimize -> we minimize negative C-index (equivalently maximize C-index).
    return -float(best_cidx)
