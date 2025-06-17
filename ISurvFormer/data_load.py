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
#    - 'id' = subject identifier
#    - 'times' = time index for longitudinal observation
#    - 'tte' = time-to-event outcome (float)
#    - 'label' = event indicator (1=event, 0=censoring)
#    - other columns = longitudinal covariates
# ========================================================
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Exclude non-feature columns
    exclude_cols = ['id', 'tte', 'times', 'label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Drop rows with missing TTE or label
    df = df.dropna(subset=['tte', 'label'])

    # Fill missing values with median (numeric) or mode (categorical)
    for col in feature_cols:
        if ptypes.is_numeric_dtype(df[col]):
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
        elif ptypes.is_object_dtype(df[col]) or ptypes.is_string_dtype(df[col]) or ptypes.is_categorical_dtype(df[col]):
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)

    # TODO：
    # The original intention here was to use masking (MASK) with zero-padding
    # and predict the missing values using LSTM.
    # Missing values include both natural missingness and padding-related missingness.
    # However, the model currently doesn't handle true missingness well,
    # so we fill missing values first to ensure training runs smoothly.
    # In the future, imputation via LSTM should be incorporated as initially planned.

    # Detect and encode string categorical variables
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

    # Group by subject ID and build time series
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

    # Padding mask: True for valid time steps
    mask_pad = torch.zeros(len(x_list), max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask_pad[i, :l] = True
        
    lengths_all = torch.tensor(lengths)
    duration_all = torch.tensor(tte_list, dtype=torch.float32)
    event_all = torch.tensor(label_list, dtype=torch.float32)

    return x_all, time_all, lengths_all, duration_all, event_all, D, mask_pad
