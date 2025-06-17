import os
import torch
import numpy as np
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
from model import DynamicSurvTransformer, generate_mask, dynamic_survival_loss_with_imputation


def train_model_cv(model_fn, optimizer_fn, alpha, max_epochs, patience,
                   x_all, x_true_all, mask_raw_all, time_all, mask_pad_all,
                   lengths_all, duration_all, event_all,
                   beta=1.0, n_splits=10, save_path=None, model_save_dir=None, vis=False):
    """
    K-Fold cross-validation training for DynamicSurvTransformer.

    Args:
        model_fn: Callable that returns a model instance.
        optimizer_fn: Callable that returns an optimizer for the model.
        alpha, beta: loss weight parameters.
        save_path: Optional path to save C-index per fold.
        model_save_dir: Optional directory to save each fold's model.
        vis: Whether to print logs.

    Returns:
        mean_cidx (float): mean concordance index across folds.
    """
    cidx_list = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    if model_save_dir is not None:
        os.makedirs(model_save_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(x_all)):
        model = model_fn()
        optimizer = optimizer_fn(model)

        # Split data
        x_train, x_val = x_all[train_idx], x_all[val_idx]
        x_true_train, x_true_val = x_true_all[train_idx], x_true_all[val_idx]
        mask_raw_train, mask_raw_val = mask_raw_all[train_idx], mask_raw_all[val_idx]
        time_train, time_val = time_all[train_idx], time_all[val_idx]
        mask_pad_train, mask_pad_val = mask_pad_all[train_idx], mask_pad_all[val_idx]
        lengths_train, lengths_val = lengths_all[train_idx], lengths_all[val_idx]
        duration_train, duration_val = duration_all[train_idx], duration_all[val_idx]
        event_train, event_val = event_all[train_idx], event_all[val_idx]

        # Train one fold
        cidx, best_model = train_one(
            model, optimizer,
            x_train, mask_raw_train, x_true_train, time_train, mask_pad_train, duration_train, event_train, lengths_train,
            x_val, mask_raw_val, x_true_val, time_val, mask_pad_val, duration_val, event_val, lengths_val,
            alpha=alpha, beta=beta, max_epochs=max_epochs, patience=patience, vis=vis
        )

        # Save model
        if model_save_dir is not None:
            model_path = os.path.join(model_save_dir, f"fold{fold+1}_model.pt")
            torch.save(best_model, model_path)
            if vis:
                print(f"\n💾 Saved model for Fold {fold+1} to: {model_path}")

        cidx_list.append(cidx)
        if vis:
            print(f"✅ Fold {fold+1}/{n_splits} | Best C-index: {cidx:.4f}")
            print('|-------------------------------------------------------------------------|')

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
              alpha=1.0, beta=1.0, max_epochs=50, patience=10, vis=False):
    """
    Train a single model with early stopping.

    Returns:
        best_cidx (float): Best validation C-index.
        model: Trained model object.
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

        if torch.isnan(x_train).any() or torch.isnan(mask_raw_train).any() or torch.isnan(t_train).any():
            if vis:
                print(f"\r⚠️ Skipping epoch {epoch} due to NaNs in training data.", end='', flush=True)
            continue

        h_seq, risk_seq, tte_seq, x_filled = model(x_train, mask_raw_train, t_train, mask_pad_train)

        if torch.isnan(h_seq).any() or torch.isnan(risk_seq).any() or torch.isnan(tte_seq).any() or torch.isnan(x_filled).any():
            if vis:
                print(f"\r⚠️ Skipping epoch {epoch} due to NaNs in model output.", end='', flush=True)
            continue

        loss = dynamic_survival_loss_with_imputation(
            h_seq, risk_seq, tte_seq, durations_train, events_train, lengths_train, times_train,
            x_filled, x_true_train, mask_raw_train, alpha=alpha, beta=beta
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            h_seq_val, risk_seq_val, _, _ = model(x_val, mask_raw_val, t_val, mask_pad_val)

            if torch.isnan(h_seq_val).any() or torch.isnan(risk_seq_val).any():
                if vis:
                    print(f"\r⚠️ Skipping epoch {epoch} due to NaNs in validation data.", end='', flush=True)
                continue

            last_idx_val = lengths_val - 1
            risk_last_val = risk_seq_val[torch.arange(len(lengths_val)), last_idx_val]

            cidx_val = concordance_index(
                durations_val.cpu().numpy(),
                -risk_last_val.cpu().numpy(),
                events_val.cpu().numpy()
            )

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

    return best_cidx, model


def compute_ibs(
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
    Compute Integrated Brier Score (IBS) on a dataset.

    Args:
        model: Trained DynamicSurvTransformer.
        x, mask_raw, time, mask_pad, lengths, durations, events: torch.Tensor.
        times_eval: Array of evaluation time points. If None, 100 points over [0, max(duration)].

    Returns:
        ibs (float): Integrated Brier Score.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    x = x.to(device); mask_raw = mask_raw.to(device)
    time = time.to(device); mask_pad = mask_pad.to(device)
    durations = durations.to(device); events = events.to(device)
    lengths = lengths.to(device)

    with torch.no_grad():
        _, _, tte_seq, _ = model(x, mask_raw, time, mask_pad)

    idx = torch.arange(len(lengths), device=device)
    last_idx = lengths - 1
    t_last = time[idx, last_idx, 0]
    tte_last = tte_seq[idx, last_idx]
    pred_event_time = (t_last + tte_last).cpu().numpy()

    durations_np = durations.cpu().numpy()
    events_np = events.cpu().numpy()

    if times_eval is None:
        max_t = durations_np.max()
        times_eval = np.linspace(0, max_t, 100)

    bs_vals = []
    for t_j in times_eval:
        mask = (durations_np > t_j) | ((durations_np <= t_j) & (events_np == 1))
        if mask.sum() == 0:
            continue
        true = (durations_np > t_j).astype(float)[mask]
        pred = (pred_event_time > t_j).astype(float)[mask]
        bs_vals.append((t_j, np.mean((true - pred) ** 2)))

    if len(bs_vals) == 0:
        return float('nan')
    times, bs = zip(*bs_vals)
    ibs = np.trapz(bs, times) / (times[-1] - times[0])
    return ibs
