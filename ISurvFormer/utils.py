import os
import time
import json
import warnings

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter

from model import dynamic_survival_loss_with_imputation


def _to_device(*tensors, device):
    out = []
    for x in tensors:
        if x is None:
            out.append(None)
        else:
            out.append(x.to(device))
    return out


def safe_concordance_index(durations, scores, events):
    durations = np.asarray(durations).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    events = np.asarray(events).reshape(-1)

    # Filter non-finite values
    m = np.isfinite(durations) & np.isfinite(scores) & np.isfinite(events)
    durations = durations[m]
    scores = scores[m]
    events = events[m].astype(int)

    if len(durations) < 2:
        return np.nan
    if np.sum(events == 1) < 1:
        return np.nan
    if len(np.unique(durations)) < 2:
        return np.nan

    try:
        return float(concordance_index(durations, scores, events))
    except Exception:
        return np.nan


def train_one_and_eval_cindex(
    model, optimizer,
    x_train, mask_raw_train, x_true_train, t_train, mask_pad_train, durations_train, events_train, lengths_train,
    x_val, mask_raw_val, x_true_val, t_val, mask_pad_val, durations_val, events_val, lengths_val,
    alpha=1.0, beta=1.0, max_epochs=50, patience=10,
    vis=False,
    return_time_to_best: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    (
        x_train, mask_raw_train, x_true_train, t_train, mask_pad_train, durations_train, events_train, lengths_train,
        x_val, mask_raw_val, x_true_val, t_val, mask_pad_val, durations_val, events_val, lengths_val,
    ) = _to_device(
        x_train, mask_raw_train, x_true_train, t_train, mask_pad_train, durations_train, events_train, lengths_train,
        x_val, mask_raw_val, x_true_val, t_val, mask_pad_val, durations_val, events_val, lengths_val,
        device=device,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    best_cidx = -1.0
    best_state = None
    time_to_best = None
    bad_epochs = 0

    for epoch in range(max_epochs):
        # --------------------
        # Training step
        # --------------------
        model.train()
        h_seq, risk_seq, tte_seq, x_filled = model(
            x_train,
            mask_raw_train,
            t_train,
            mask_pad_train,
        )
        times_train = t_train.squeeze(-1)

        loss = dynamic_survival_loss_with_imputation(
            h_seq,
            risk_seq,
            tte_seq,
            durations_train,
            events_train,
            lengths_train,
            times_train,
            x_filled,
            x_true_train,
            mask_raw_train,
            alpha=alpha,
            beta=beta,
        )

        # Skip update if loss becomes non-finite
        if not torch.isfinite(loss):
            if vis:
                print(
                    f"\rEpoch {epoch + 1:03d} | Loss NaN/Inf -> skip",
                    end="",
                    flush=True,
                )
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # --------------------
        # Validation step
        # --------------------
        model.eval()
        with torch.no_grad():
            _, risk_seq_val, _, _ = model(
                x_val,
                mask_raw_val,
                t_val,
                mask_pad_val,
            )

            last_idx_val = lengths_val - 1
            idx = torch.arange(len(lengths_val), device=device)
            risk_last_val = risk_seq_val[idx, last_idx_val]

            # Skip evaluation if risk contains NaN/Inf
            if not torch.isfinite(risk_last_val).all():
                if vis:
                    print(
                        f"\rEpoch {epoch + 1:03d} | risk NaN/Inf -> skip eval",
                        end="",
                        flush=True,
                    )
                continue

            cidx_val = safe_concordance_index(
                durations_val.detach().cpu().numpy(),
                (-risk_last_val).detach().cpu().numpy(),
                events_val.detach().cpu().numpy(),
            )

        # Skip if C-index is invalid (e.g., no comparable pairs)
        if not np.isfinite(cidx_val):
            if vis:
                print(
                    f"\rEpoch {epoch + 1:03d} | C-index invalid -> skip",
                    end="",
                    flush=True,
                )
            continue

        if vis:
            print(
                (
                    f"\rEpoch {epoch + 1:03d} | Loss: {loss.item():.4f} "
                    f"| Val C-index: {cidx_val:.4f} "
                    f"| Best: {max(best_cidx, cidx_val):.4f}"
                ),
                end="",
                flush=True,
            )

        # Update best model state
        if cidx_val > best_cidx:
            best_cidx = float(cidx_val)
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time_to_best = time.perf_counter() - t0
            bad_epochs = 0
        else:
            bad_epochs += 1

        # Early stopping
        if bad_epochs >= patience:
            if vis:
                print(
                    f"\nEarly stopping at epoch {epoch + 1} "
                    f"| Best C-index: {best_cidx:.4f}"
                )
            break

    if vis:
        print()

    # If training never produced a valid C-index, return NaN and let caller handle pruning
    if best_cidx < 0:
        return np.nan, None

    if time_to_best is None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_to_best = time.perf_counter() - t0

    if return_time_to_best:
        return best_cidx, best_state, float(time_to_best)
    else:
        return best_cidx, best_state


def compute_ipcw_weights(durations, events, times_eval):
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)

    censor = 1 - events
    km = KaplanMeierFitter()
    km.fit(durations, event_observed=censor)

    def G(x):
        """Helper to evaluate G(x) with a numerical floor."""
        g = float(km.predict(x))
        return max(g, 1e-6)

    G_t = np.array([G(t) for t in times_eval], dtype=float)
    G_T = np.array([G(ti) for ti in durations], dtype=float)

    return G_t, G_T


def compute_ipcw_ibs_from_pred_survival(durations, events, pred_surv_indicator, times_eval):
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    times_eval = np.asarray(times_eval, dtype=float)

    G_t, G_T = compute_ipcw_weights(durations, events, times_eval)

    bs = []
    for j, t in enumerate(times_eval):
        # True survival indicator I(T > t)
        y = (durations > t).astype(float)

        # IPCW weights
        w = np.zeros_like(durations, dtype=float)

        # Events that occurred by time t
        mask_event_by_t = (durations <= t) & (events == 1)
        w[mask_event_by_t] = 1.0 / G_T[mask_event_by_t]

        # Subjects still at risk at time t
        mask_at_risk = durations > t
        w[mask_at_risk] = 1.0 / G_t[j]

        # Censored before t have weight zero (implicitly)

        if w.sum() <= 0:
            continue

        p = pred_surv_indicator[:, j].astype(float)
        bs_t = np.sum(w * (y - p) ** 2) / np.sum(w)
        bs.append(bs_t)

    if len(bs) < 2:
        return float("nan")

    ibs = np.trapz(bs, times_eval[: len(bs)]) / (
        times_eval[len(bs) - 1] - times_eval[0]
    )
    return float(ibs)


def compute_ipcw_ibs(
    model,
    x,
    mask_raw,
    time,
    mask_pad,
    lengths,
    durations,
    events,
    times_eval=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    x = x.to(device)
    mask_raw = mask_raw.to(device)
    time = time.to(device)
    mask_pad = mask_pad.to(device)
    lengths = lengths.to(device)
    durations_t = durations.to(device)
    events_t = events.to(device)

    with torch.no_grad():
        _, _, tte_seq, _ = model(x, mask_raw, time, mask_pad)

    idx = torch.arange(len(lengths), device=device)
    last_idx = lengths - 1
    t_last = time[idx, last_idx, 0]
    tte_last = tte_seq[idx, last_idx]
    pred_event_time = (t_last + tte_last).detach().cpu().numpy()

    durations_np = durations_t.detach().cpu().numpy().astype(float)
    events_np = events_t.detach().cpu().numpy().astype(int)

    if times_eval is None:
        max_t = float(np.max(durations_np))
        times_eval = np.linspace(0.0, max_t, 100)

    # Predicted survival indicator/probability
    pred_surv = (pred_event_time[:, None] > times_eval[None, :]).astype(float)

    return compute_ipcw_ibs_from_pred_survival(
        durations_np,
        events_np,
        pred_surv,
        times_eval,
    )

def train_model_cv(
    model_fn, optimizer_fn,
    alpha, beta,
    max_epochs, patience,
    x_all, x_true_all,
    mask_raw_all, time_all, mask_pad_all, lengths_all, duration_all, event_all,
    n_splits=10,
    save_path=None, model_save_dir=None, split_save_path=None, ibs_save_path=None, time_save_path=None,
    vis=False,
):
    def _to_1d_numpy(a):
        """Convert torch tensor or array-like to 1D numpy array."""
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        return np.asarray(a).reshape(-1)

    cidx_list = []
    ibs_list = []
    splits_dump = []
    time_to_best_list = []

    # =========================================================
    # Stratified CV by event/censoring status (preferred)
    # =========================================================
    y = _to_1d_numpy(event_all).astype(int)

    n_event = int((y == 1).sum())
    n_cens = int((y == 0).sum())

    if min(n_event, n_cens) >= n_splits:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        )
        split_iter = splitter.split(x_all, y)
        if vis:
            print(
                "[CV] Using StratifiedKFold by event status "
                f"| events={n_event}, censored={n_cens}"
            )
    else:
        # If one class is too small, StratifiedKFold may fail. Fall back to standard KFold.
        warnings.warn(
            "[CV] Not enough samples per class for StratifiedKFold "
            f"(events={n_event}, censored={n_cens}, n_splits={n_splits}). "
            "Falling back to standard KFold (not stratified)."
        )
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        )
        split_iter = splitter.split(x_all)

    if model_save_dir is not None:
        os.makedirs(model_save_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(split_iter, start=1):
        if vis:
            print(f"\n[CV] Fold {fold}/{n_splits}")

        model = model_fn()
        optimizer = optimizer_fn(model)

        # Split data for this fold
        x_tr, x_va = x_all[train_idx], x_all[val_idx]
        x_true_tr, x_true_va = x_true_all[train_idx], x_true_all[val_idx]
        mask_raw_tr, mask_raw_va = mask_raw_all[train_idx], mask_raw_all[val_idx]
        time_tr, time_va = time_all[train_idx], time_all[val_idx]
        mask_pad_tr, mask_pad_va = mask_pad_all[train_idx], mask_pad_all[val_idx]
        lengths_tr, lengths_va = lengths_all[train_idx], lengths_all[val_idx]
        dur_tr, dur_va = duration_all[train_idx], duration_all[val_idx]
        evt_tr, evt_va = event_all[train_idx], event_all[val_idx]

        # Train and get best state
        best_cidx, best_state, best_time_to_reach = train_one_and_eval_cindex(
            model=model,
            optimizer=optimizer,
            x_train=x_tr,
            mask_raw_train=mask_raw_tr,
            x_true_train=x_true_tr,
            t_train=time_tr,
            mask_pad_train=mask_pad_tr,
            durations_train=dur_tr,
            events_train=evt_tr,
            lengths_train=lengths_tr,
            x_val=x_va,
            mask_raw_val=mask_raw_va,
            x_true_val=x_true_va,
            t_val=time_va,
            mask_pad_val=mask_pad_va,
            durations_val=dur_va,
            events_val=evt_va,
            lengths_val=lengths_va,
            alpha=alpha,
            beta=beta,
            max_epochs=max_epochs,
            patience=patience,
            vis=vis,
            return_time_to_best=True,
        )
        time_to_best_list.append(float(best_time_to_reach))

        # Load best state into model for saving and IBS computation
        if best_state is not None:
            model.load_state_dict(best_state)

        # Save model at best epoch
        if model_save_dir is not None:
            model_path = os.path.join(model_save_dir, f"fold{fold}_model.pt")
            # Save full model object to match downstream usage
            torch.save(model, model_path)
            if vis:
                print(f"[CV] Saved best model for fold {fold} to: {model_path}")

        # Compute IPCW-IBS on validation split (out-of-fold)
        fold_ibs = compute_ipcw_ibs(
            model=model,
            x=x_va,
            mask_raw=mask_raw_va,
            time=time_va,
            mask_pad=mask_pad_va,
            lengths=lengths_va,
            durations=dur_va,
            events=evt_va,
            times_eval=None,
        )

        cidx_list.append(best_cidx)
        ibs_list.append(fold_ibs)

        splits_dump.append(
            {
                "fold": fold,
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
            }
        )

        if vis:
            print(
                f"[CV] Fold {fold} | Best C-index (val): {best_cidx:.4f} "
                f"| IPCW-IBS (val): {fold_ibs:.4f}"
            )
            print("|" + "-" * 73 + "|")

    # Save C-index results
    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            for i, c in enumerate(cidx_list, start=1):
                f.write(f"Fold {i}: C-index = {c:.4f}\n")
            f.write(f"\nMean C-index = {np.mean(cidx_list):.4f}\n")
            f.write(f"Std  C-index = {np.std(cidx_list):.4f}\n")

    # Save IBS results
    if ibs_save_path is not None:
        with open(ibs_save_path, "w", encoding="utf-8") as f:
            for i, b in enumerate(ibs_list, start=1):
                f.write(f"Fold {i}: IPCW-IBS = {b:.4f}\n")
            f.write(f"\nMean IPCW-IBS = {np.mean(ibs_list):.4f}\n")
            f.write(f"Std  IPCW-IBS = {np.std(ibs_list):.4f}\n")

    # Save time-to-best results
    if time_save_path is not None:
        with open(time_save_path, "w", encoding="utf-8") as f:
            for i, tsec in enumerate(time_to_best_list, start=1):
                f.write(f"Fold {i}: Time-to-best (sec) = {tsec:.4f}\n")
            f.write(
                f"\nMean Time-to-best (sec) = {np.mean(time_to_best_list):.4f}\n"
            )
            f.write(
                f"Std  Time-to-best (sec) = {np.std(time_to_best_list):.4f}\n"
            )

    # Save splits (train/val indices)
    if split_save_path is not None:
        with open(split_save_path, "w", encoding="utf-8") as f:
            json.dump(splits_dump, f, indent=2)

    return float(np.mean(cidx_list)), float(np.mean(ibs_list))
