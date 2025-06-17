import optuna
import torch
import numpy as np
from model import DynamicSurvTransformer
from utils import train_model_cv

def objective(trial, D, x_all, x_true_all, mask_raw_all, time_all, mask_pad_all,
              lengths_all, duration_all, event_all):
    d_model = trial.suggest_categorical('d_model', [64, 128, 256])
    nhead = trial.suggest_int('nhead', 1, 8)
    if d_model % nhead != 0:
        raise optuna.exceptions.TrialPruned()
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    mlp_hidden_ratio = trial.suggest_categorical('mlp_hidden_ratio', [0.5, 1.0, 2.0])
    mlp_num_layers = trial.suggest_int('mlp_num_layers', 1, 4)
    activation = trial.suggest_categorical('activation', ['ReLU', 'GELU', 'SELU', 'LeakyReLU'])
    risk_hidden_dim = trial.suggest_categorical("risk_hidden_dim", [64, 128, 256])
    risk_num_layers = trial.suggest_int("risk_num_layers", 2, 4)
    risk_activation = trial.suggest_categorical("risk_activation", ['ReLU', 'LeakyReLU', 'SELU', 'GELU'])
    alpha = trial.suggest_float("alpha", 0.1, 5.0, log=True)
    beta = trial.suggest_float("beta", 0.1, 5.0, log=True)

    def model_fn():
        return DynamicSurvTransformer(
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
            risk_activation=risk_activation
        )

    def optimizer_fn(model):
        return torch.optim.Adam(model.parameters(), lr=lr)

    mean_cidx = train_model_cv(model_fn, optimizer_fn,
                               alpha=alpha, beta=beta, max_epochs=100, patience=20,
                               x_all=x_all, x_true_all=x_true_all, mask_raw_all=mask_raw_all,
                               time_all=time_all, mask_pad_all=mask_pad_all,
                               lengths_all=lengths_all, duration_all=duration_all, event_all=event_all,
                               n_splits=10,vis=False)
    if np.isnan(mean_cidx):
        raise RuntimeError("NaN occurred in C-index. Skipping this trial.")
    return -mean_cidx
