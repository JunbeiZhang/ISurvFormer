import numpy as np
import pandas as pd
import torch
import pandas.api.types as ptypes  # is_numeric_dtype / is_bool_dtype / is_datetime64_any_dtype

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

        if ptypes.is_numeric_dtype(s):
            df[col] = pd.to_numeric(s, errors="coerce")
            continue

        if ptypes.is_bool_dtype(s):
            df[col] = s.astype("float32")
            continue

        if ptypes.is_datetime64_any_dtype(s):
            ss = pd.to_datetime(s, errors="coerce")
            vals = ss.view("int64")  
            vals = pd.Series(vals, index=ss.index, dtype="float64")
            vals[ss.isna()] = np.nan
            df[col] = (vals / 1e9).astype("float32")
            continue

        cat = s.astype("category")
        codes = pd.Series(cat.cat.codes, index=s.index).astype("float32")  
        codes = codes.mask(codes < 0, np.nan)  
        df[col] = codes

def load_data(csv_path):
    df = pd.read_csv(csv_path)

    exclude_cols = ['id', 'tte', 'times', 'label']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    df = df.dropna(subset=['tte', 'label'])

    _numericize_features_inplace(df, feature_cols)

    grouped = df.groupby('id')
    x_list, t_list, mask_raw_list = [], [], []
    tte_list, label_list, lengths = [], [], []

    for _, g in grouped:
        g = g.sort_values('times')
        x = g[feature_cols].astype('float32').to_numpy()  
        t = g[['times']].astype('float32').to_numpy()

        mask_raw = ~np.isnan(x)

        x_list.append(torch.tensor(x))
        t_list.append(torch.tensor(t))
        mask_raw_list.append(torch.tensor(mask_raw, dtype=torch.bool))

        tte_list.append(g['tte'].values[0])
        label_list.append(g['label'].values[0])
        lengths.append(len(g))

    max_len = max(lengths)
    D = len(feature_cols)

    x_all    = pad_tensor_list(x_list, max_len, D)
    time_all = pad_tensor_list(t_list, max_len, 1)

    mask_pad = torch.zeros(len(x_list), max_len, dtype=torch.bool)
    for i, L in enumerate(lengths):
        mask_pad[i, :L] = True

    mask_raw_all = pad_bool_list(mask_raw_list, max_len, D)
    mask_raw_all[~mask_pad.unsqueeze(-1).expand_as(mask_raw_all)] = False

    lengths_all  = torch.tensor(lengths)
    duration_all = torch.tensor(tte_list, dtype=torch.float32)
    event_all    = torch.tensor(label_list, dtype=torch.float32)

    return x_all, time_all, lengths_all, duration_all, event_all, D, mask_pad, mask_raw_all
