import os
import torch
import numpy as np
from sklearn.model_selection import KFold
from model import dynamic_survival_loss_with_imputation
from lifelines.utils import concordance_index

def train_model_cv(model_fn, optimizer_fn, alpha, max_epochs, patience,
                   x_all, x_true_all, mask_raw_all, time_all, mask_pad_all,
                   lengths_all, duration_all, event_all,
                   beta=1.0, gamma=1.0, n_splits=10, save_path=None, model_save_dir=None, vis=False):
    """
    Perform K-fold cross-validation training for dynamic survival models with imputation.
    """
    cidx_list = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    if model_save_dir is not None:
        os.makedirs(model_save_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(x_all)):
        model = model_fn()
        optimizer = optimizer_fn(model)

        # Split training and validation sets
        x_train, x_val = x_all[train_idx], x_all[val_idx]
        x_true_train, x_true_val = x_true_all[train_idx], x_true_all[val_idx]
        mask_raw_train, mask_raw_val = mask_raw_all[train_idx], mask_raw_all[val_idx]
        time_train, time_val = time_all[train_idx], time_all[val_idx]
        mask_pad_train, mask_pad_val = mask_pad_all[train_idx], mask_pad_all[val_idx]
        lengths_train, lengths_val = lengths_all[train_idx], lengths_all[val_idx]
        duration_train, duration_val = duration_all[train_idx], duration_all[val_idx]
        event_train, event_val = event_all[train_idx], event_all[val_idx]

        # Train and evaluate the model
        cidx, best_model = train_one(
            model, optimizer,
            x_train, mask_raw_train, x_true_train, time_train, mask_pad_train, duration_train, event_train, lengths_train,
            x_val, mask_raw_val, x_true_val, time_val, mask_pad_val, duration_val, event_val, lengths_val,
            alpha=alpha, beta=beta, gamma=gamma, max_epochs=max_epochs, patience=patience, vis=vis
        )

        # Save best model weights
        if model_save_dir is not None:
            model_path = os.path.join(model_save_dir, f"fold{fold+1}_model.pt")
            torch.save(best_model.state_dict(), model_path)
            if vis:
                print(f"\n💾 Saved model for Fold {fold+1} to: {model_path}")

        cidx_list.append(cidx)
        if vis:
            print(f"✅ Fold {fold+1}/{n_splits} | Best C-index: {cidx:.4f}")
            print('|-------------------------------------------------------------------------|')

    # Save C-index results
    if save_path is not None:
        with open(save_path, 'w') as f:
            for i, c in enumerate(cidx_list):
                f.write(f"Fold {i+1}: C-index = {c:.4f}\n")
            f.write(f"\nMean C-index = {np.mean(cidx_list):.4f}\n")
            f.write(f"Std  C-index = {np.std(cidx_list):.4f}\n")

    return np.mean(cidx_list)


def train_one(model, optimizer,
              x_train, mask_raw_train, x_true_train, t_train, mask_pad_train, durations_train, events_train, lengths_train,
              x_val, mask_raw_val, x_true_val, t_val, mask_pad_val, durations_val, events_val, lengths_val,
              alpha=1.0, beta=1.0, gamma=1.0, max_epochs=50, patience=10, vis=False):
    """
    Train the model for one fold with early stopping.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Move data to device
    x_train, x_true_train, mask_raw_train = x_train.to(device), x_true_train.to(device), mask_raw_train.to(device)
    t_train, mask_pad_train = t_train.to(device), mask_pad_train.to(device)
    durations_train, events_train, lengths_train = durations_train.to(device), events_train.to(device), lengths_train.to(device)

    x_val, x_true_val, mask_raw_val = x_val.to(device), x_true_val.to(device), mask_raw_val.to(device)
    t_val, mask_pad_val = t_val.to(device), mask_pad_val.to(device)
    durations_val, events_val, lengths_val = durations_val.to(device), events_val.to(device), lengths_val.to(device)

    best_cidx = 0.0
    best_model_state = None
    bad_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        times_train = t_train.squeeze(-1)

        # Skip epoch if training data contains NaNs
        if torch.isnan(x_train).any() or torch.isnan(mask_raw_train).any() or torch.isnan(t_train).any():
            if vis:
                print(f"\rSkipping epoch {epoch} due to NaNs in training data.", end='', flush=True)
            continue

        h_seq, eta, tte_seq, x_filled, q_probs = model(x_train, mask_raw_train, t_train, mask_pad_train)

        if torch.isnan(h_seq).any() or torch.isnan(eta).any() or torch.isnan(tte_seq).any() or torch.isnan(x_filled).any():
            if vis:
                print(f"\rSkipping epoch {epoch} due to NaNs in model output.", end='', flush=True)
            continue

        loss = dynamic_survival_loss_with_imputation(
            q_probs, eta, tte_seq, durations_train, events_train, lengths_train, times_train,
            x_filled, x_true_train, mask_raw_train,
            alpha=alpha, beta=beta, gamma=gamma
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            _, risk_seq_val, _, _, _ = model(x_val, mask_raw_val, t_val, mask_pad_val)
            last_idx_val = lengths_val - 1
            eta_val = risk_seq_val[torch.arange(len(lengths_val)), last_idx_val]

            if torch.isnan(eta_val).any():
                if vis:
                    print(f"\rSkipping epoch {epoch} due to NaNs in validation data.", end='', flush=True)
                continue

            cidx_val = concordance_index(
                durations_val.squeeze(-1).cpu().numpy(),
                -eta_val.cpu().numpy(),
                events_val.squeeze(-1).cpu().numpy())

        if cidx_val > best_cidx:
            best_cidx = cidx_val
            best_model_state = model.state_dict()
            bad_epochs = 0
        else:
            bad_epochs += 1

        if vis:
            print(f"\rEpoch {epoch+1:03d} | Loss: {loss.item():.2f} | Val C-index: {cidx_val:.4f} | Best: {best_cidx:.4f}", end='', flush=True)

        if bad_epochs >= patience:
            if vis:
                print(f"\n🛑 Early stopping at epoch {epoch+1} with best C-index = {best_cidx:.4f}")
            break

    model.load_state_dict(best_model_state)
    return best_cidx, model


from sksurv.metrics import brier_score
from sksurv.util import Surv

def compute_ibs_from_risk(
    model,
    x,
    mask_raw,
    time,
    mask_pad,
    lengths,
    durations,
    events,
    times_eval: np.ndarray = None
) -> float:
    """
    Compute integrated Brier score (IBS) based on model-predicted risk sequence.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    x = x.to(device); mask_raw = mask_raw.to(device)
    time = time.to(device); mask_pad = mask_pad.to(device)
    durations = durations.to(device); events = events.to(device)
    lengths = lengths.to(device)

    with torch.no_grad():
        _, risk_seq, _, _, _ = model(x, mask_raw, time, mask_pad)  # [B, T]

    time_np = time.cpu().numpy().squeeze(-1)
    risk_seq_np = risk_seq.cpu().numpy()
    durations_np = durations.cpu().numpy()
    events_np = events.cpu().numpy()

    max_t = durations_np.max()
    if times_eval is None:
        times_eval = np.linspace(0, max_t, 100)

    bs_vals = []
    for t_j in times_eval:
        time_diff = np.abs(time_np - t_j)
        idxs = time_diff.argmin(axis=1)

        pred = np.array([risk_seq_np[b, idxs[b]] for b in range(risk_seq_np.shape[0])])
        mask = (durations_np > t_j) | ((durations_np <= t_j) & (events_np == 1))
        if mask.sum() == 0:
            continue

        true = (durations_np > t_j).astype(float)[mask]
        pred_masked = (pred < np.median(pred)).astype(float)[mask]
        bs_vals.append((t_j, np.mean((true - pred_masked) ** 2)))

    if len(bs_vals) == 0:
        return float('nan')
    times, bs = zip(*bs_vals)
    ibs = np.trapz(bs, times) / (times[-1] - times[0])
    return ibs
