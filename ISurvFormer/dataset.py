import numpy as np
import pandas as pd
import torch
import pandas.api.types as ptypes


def pad_tensor_list(tensor_list, max_len, dim):
    padded = torch.zeros(len(tensor_list), max_len, dim)
    for i, t in enumerate(tensor_list):
        L = t.size(0)
        padded[i, :L, :] = t
    return padded


def pad_bool_list(mask_list, max_len, dim):
    out = torch.zeros(len(mask_list), max_len, dim, dtype=torch.bool)
    for i, m in enumerate(mask_list):
        L = m.size(0)
        out[i, :L, :] = m
    return out


def _numericize_features_inplace(df: pd.DataFrame, feature_cols):
    for col in feature_cols:
        s = df[col]

        # Already numeric (int/float), but may contain mixed types (e.g., strings).
        # Coerce non-numeric values to NaN.
        if ptypes.is_numeric_dtype(s):
            df[col] = pd.to_numeric(s, errors="coerce")
            continue

        # Boolean column: convert to 0/1, missing remains NaN.
        if ptypes.is_bool_dtype(s):
            # astype(float32) converts True/False -> 1.0/0.0 and keeps NaN as NaN.
            df[col] = s.astype("float32")
            continue

        # Datetime column: convert to Unix timestamp in seconds (float), invalid -> NaN.
        if ptypes.is_datetime64_any_dtype(s):
            ss = pd.to_datetime(s, errors="coerce")
            # astype('int64') gives nanosecond timestamps; convert to seconds.
            vals = ss.view("int64")  # NaT becomes the minimum int64; handle with mask.
            vals = pd.Series(vals, index=ss.index, dtype="float64")
            vals[ss.isna()] = np.nan
            df[col] = (vals / 1e9).astype("float32")
            continue

        # Other types (object/category, etc.): encode as categorical codes, missing as NaN.
        cat = s.astype("category")
        codes = pd.Series(cat.cat.codes, index=s.index).astype("float32")  # -1 indicates NaN
        codes = codes.mask(codes < 0, np.nan)  # convert -1 back to NaN
        df[col] = codes


def load_data(csv_path):
    df = pd.read_csv(csv_path)

    exclude_cols = ["id", "tte", "times", "label"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Only drop samples with missing tte/label.
    # Other feature missing values are kept as NaN for imputation learning.
    df = df.dropna(subset=["tte", "label"])

    # Convert feature columns to numeric (in-place).
    _numericize_features_inplace(df, feature_cols)

    # Group by subject and align sequences.
    grouped = df.groupby("id")
    x_list, t_list, mask_raw_list = [], [], []
    tte_list, label_list, lengths = [], [], []

    for _, g in grouped:
        g = g.sort_values("times")

        # At this point df[feature_cols] is already numeric float/NaN.
        x = g[feature_cols].astype("float32").to_numpy()  # allow NaN
        t = g[["times"]].astype("float32").to_numpy()

        # Observation mask: True = observed, False = missing.
        # Padding positions will be set to False later.
        mask_raw = ~np.isnan(x)

        x_list.append(torch.tensor(x))
        t_list.append(torch.tensor(t))
        mask_raw_list.append(torch.tensor(mask_raw, dtype=torch.bool))

        tte_list.append(g["tte"].values[0])
        label_list.append(g["label"].values[0])
        lengths.append(len(g))

    max_len = max(lengths)
    D = len(feature_cols)

    # Pad sequences in time dimension.
    # True missing entries remain NaN; padded entries are filled with 0.
    x_all = pad_tensor_list(x_list, max_len, D)
    time_all = pad_tensor_list(t_list, max_len, 1)

    # Time padding mask: True = valid time step, False = padding.
    mask_pad = torch.zeros(len(x_list), max_len, dtype=torch.bool)
    for i, L in enumerate(lengths):
        mask_pad[i, :L] = True

    # 3D observation mask; enforce padding positions as False.
    mask_raw_all = pad_bool_list(mask_raw_list, max_len, D)
    mask_raw_all[~mask_pad.unsqueeze(-1).expand_as(mask_raw_all)] = False

    lengths_all = torch.tensor(lengths)
    duration_all = torch.tensor(tte_list, dtype=torch.float32)
    event_all = torch.tensor(label_list, dtype=torch.float32)

    # Keep the original return interface.
    return x_all, time_all, lengths_all, duration_all, event_all, D, mask_pad, mask_raw_all
