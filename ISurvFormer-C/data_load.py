import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import pandas.api.types as ptypes


def pad_tensor_list(tensor_list, max_len, dim):
    """
    Pad a list of 2D tensors to the same length along time dimension.

    Args:
        tensor_list (List[Tensor]): list of [T_i, D] tensors
        max_len (int): target maximum length
        dim (int): feature dimension

    Returns:
        Tensor: [B, max_len, D]
    """
    padded = torch.zeros(len(tensor_list), max_len, dim)
    for i, t in enumerate(tensor_list):
        padded[i, :t.size(0), :] = t
    return padded


def load_data(csv_path):
    """
    Load longitudinal data for survival modeling.

    Expected columns:
        - 'id', 'times': identify subject and observation time
        - 'tte': time-to-event
        - 'label': event indicator (1=event, 0=censored)
        - other columns are treated as features
    """
    df = pd.read_csv(csv_path)

    exclude_cols = ['id', 'tte', 'times', 'label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Drop rows with missing time-to-event or event label
    df = df.dropna(subset=['tte', 'label'])

    # Fill missing values with median (numerical) or mode (categorical)
    for col in feature_cols:
        if ptypes.is_numeric_dtype(df[col]):
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
        elif ptypes.is_object_dtype(df[col]) or ptypes.is_string_dtype(df[col]) or ptypes.is_categorical_dtype(df[col]):
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)

    # TODO：
    # The original plan was to use LSTM to impute missing values (including natural missingness
    # and padding-induced missingness). However, current model does not support raw missing NaNs.
    # So median/mode imputation is used here to ensure model runs.
    # Future work should add LSTM-based imputation.

    # Detect and encode string categorical features
    categorical_cols = []
    label_encoders = {}
    categorical_info = {}

    for col in feature_cols:
        if ptypes.is_object_dtype(df[col]) or ptypes.is_string_dtype(df[col]) or ptypes.is_categorical_dtype(df[col]):
            unique_values = df[col].nunique(dropna=True)
            if unique_values <= 1:
                print(f"⚠️ Skipping categorical column '{col}' (only {unique_values} unique value).")
                continue

            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            categorical_info[col] = len(le.classes_)
            categorical_cols.append(col)

    print(f"🧠 Detected {len(categorical_cols)} categorical columns: {categorical_cols}")
    print(f"🧠 After encoding, NaN values in categorical columns: {df[categorical_cols].isna().sum().sum()}")

    # Group data by subject (id)
    grouped = df.groupby('id')

    x_list, t_list, tte_list, label_list, lengths = [], [], [], [], []

    for _, group in grouped:
        group_sorted = group.sort_values("times")
        x = group_sorted[feature_cols].values.astype('float32')
        t = group_sorted[["times"]].values.astype('float32')
        tte = group_sorted["tte"].values[0]
        label = group_sorted["label"].values[0]

        x_list.append(torch.tensor(x))
        t_list.append(torch.tensor(t))
        tte_list.append(tte)
        label_list.append(label)
        lengths.append(len(group))

    max_len = max(lengths)
    D = len(feature_cols)

    x_all = pad_tensor_list(x_list, max_len, D)
    time_all = pad_tensor_list(t_list, max_len, 1)

    mask_pad = torch.zeros(len(x_list), max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask_pad[i, :l] = True

    lengths_all = torch.tensor(lengths)
    duration_all = torch.tensor(tte_list, dtype=torch.float32)
    event_all = torch.tensor(label_list, dtype=torch.float32)

    return x_all, time_all, lengths_all, duration_all, event_all, D, mask_pad
