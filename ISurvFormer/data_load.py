import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import pandas.api.types as ptypes


def pad_tensor_list(tensor_list, max_len, dim):
    """Pad a list of tensors to the same length with zeros."""
    padded = torch.zeros(len(tensor_list), max_len, dim)
    for i, t in enumerate(tensor_list):
        padded[i, :t.size(0), :] = t
    return padded


# ========================================================
# Expected input CSV format:
#    - One row = one observation at a specific timepoint
#    - Required columns: 'id', 'times', 'tte', 'label'
#    - other columns = longitudinal covariates
# ========================================================
def load_data(csv_path, mask_ratio=0.15, random_state=42):
    """
    Self-supervised imputation data loader (masking + padding).

    Returns:
        x_raw_all      : [B, T, D] input with (natural missing + artificial masked + pad) set to 0
        x_true_all     : [B, T, D] original values (NaN filled 0 only for tensor storage)
        mask_raw_all   : [B, T, D] 1=visible/observed (and NOT artificially masked), 0=missing/masked/pad
        mask_train_all : [B, T, D] 1=artificially masked (supervision targets), 0=otherwise
        time_all       : [B, T, 1]
        lengths_all    : [B]
        duration_all   : [B]
        event_all      : [B]
        D              : int
        mask_pad       : [B, T] True=valid time steps, False=pad
        feature_cols   : list[str]
        label_encoders : dict[col -> LabelEncoder]
        categorical_info: dict[col -> n_classes]
    """
    # ---- RNG (reproducible masking) ----
    rng = torch.Generator()
    rng.manual_seed(int(random_state))

    df = pd.read_csv(csv_path)

    # ---- columns ----
    exclude_cols = ['id', 'tte', 'times', 'label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Drop rows with missing survival labels
    df = df.dropna(subset=['tte', 'label'])

    # ---- detect categorical feature columns ----
    categorical_cols = []
    for col in feature_cols:
        if (ptypes.is_object_dtype(df[col]) or ptypes.is_string_dtype(df[col]) or
                ptypes.is_categorical_dtype(df[col])):
            categorical_cols.append(col)

    # ---- fit LabelEncoders with explicit missing token ----
    label_encoders = {}
    categorical_info = {}
    for col in categorical_cols:
        nunq = df[col].nunique(dropna=True)
        if nunq <= 1:
            print(f"⚠️ Skipping categorical column '{col}' (only {nunq} unique value).")
            continue

        le = LabelEncoder()
        series = df[col].copy()
        series = series.where(~series.isna(), "__MISSING__").astype(str)
        le.fit(series.values)

        label_encoders[col] = le
        categorical_info[col] = len(le.classes_)

    # ---- apply encoding (keep natural-missing info via obs_mask later) ----
    for col in categorical_cols:
        if col not in label_encoders:
            continue
        le = label_encoders[col]
        series = df[col].copy()
        series = series.where(~series.isna(), "__MISSING__").astype(str)
        df[col] = le.transform(series.values).astype("float32")

    print(f" Encoded categorical columns: {[c for c in categorical_cols if c in label_encoders]}")
    if categorical_info:
        print(f" categorical_info (n_classes): {categorical_info}")

    # ---- group by subject id ----
    grouped = df.groupby('id')

    x_raw_list, x_true_list = [], []
    mask_raw_list, mask_train_list = [], []
    t_list, tte_list, label_list, lengths = [], [], [], []

    for _, group in grouped:
        group_sorted = group.sort_values("times")
        X_df = group_sorted[feature_cols].copy()

        # obs mask based on ORIGINAL NaN (before we fill)
        obs_mask_df = ~X_df.isna()                 # True=observed, False=natural missing

        # ensure numeric columns are numeric
        for col in feature_cols:
            if col in label_encoders:
                # already encoded (float32); original NaN is tracked by obs_mask_df
                continue
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

        # tensors
        X_true = X_df.fillna(0.0).values.astype("float32")      # [L, D]
        obs_mask = obs_mask_df.values.astype("float32")         # [L, D] 1=observed, 0=natural missing

        L = X_true.shape[0]
        D = X_true.shape[1]

        obs_mask_t = torch.tensor(obs_mask, dtype=torch.float32)

        # artificial masking: only on observed entries
        bern = torch.rand((L, D), generator=rng)
        mask_train = ((bern < float(mask_ratio)) & (obs_mask_t == 1)).to(torch.float32)  # 1=masked target

        # raw/visible mask: observed and not artificially masked
        mask_raw = (obs_mask_t * (1.0 - mask_train)).to(torch.float32)  # 1=visible, 0=missing/masked

        x_true_t = torch.tensor(X_true, dtype=torch.float32)
        x_raw_t = x_true_t.clone()
        x_raw_t[mask_raw == 0] = 0.0  # hide natural missing + artificial masked

        # time/survival labels
        t = group_sorted[["times"]].values.astype("float32")     # [L, 1]
        tte = float(group_sorted["tte"].values[0])
        label = float(group_sorted["label"].values[0])

        x_raw_list.append(x_raw_t)
        x_true_list.append(x_true_t)
        mask_raw_list.append(mask_raw)
        mask_train_list.append(mask_train)

        t_list.append(torch.tensor(t, dtype=torch.float32))
        tte_list.append(tte)
        label_list.append(label)
        lengths.append(L)

    # ---- padding ----
    max_len = max(lengths)
    D = len(feature_cols)

    x_raw_all = pad_tensor_list(x_raw_list, max_len, D)
    x_true_all = pad_tensor_list(x_true_list, max_len, D)
    mask_raw_all = pad_tensor_list(mask_raw_list, max_len, D)
    mask_train_all = pad_tensor_list(mask_train_list, max_len, D)
    time_all = pad_tensor_list(t_list, max_len, 1)

    # mask_pad: True for valid time steps
    mask_pad = torch.zeros(len(x_raw_list), max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask_pad[i, :l] = True

    # force padded region to 0
    pad_b = (~mask_pad).unsqueeze(-1).expand(-1, -1, D)
    x_raw_all[pad_b] = 0.0
    x_true_all[pad_b] = 0.0
    mask_raw_all[pad_b] = 0.0
    mask_train_all[pad_b] = 0.0

    lengths_all = torch.tensor(lengths, dtype=torch.long)
    duration_all = torch.tensor(tte_list, dtype=torch.float32)
    event_all = torch.tensor(label_list, dtype=torch.float32)

    return (
        x_raw_all, x_true_all, mask_raw_all, mask_train_all,
        time_all, lengths_all, duration_all, event_all,
        D, mask_pad
    )
