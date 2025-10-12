import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def build_vectors(df, time_col, time_cols, target_col, lag_value, target_lag=None):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col, ascending=True).reset_index(drop=True)
    target_lag = target_lag or lag_value

    values = df[target_col].to_numpy(float)
    features = [c for c in df.columns if c not in [time_col, target_col] + time_cols]

    time_features = df[time_cols].to_numpy(float)
    time_scaled = MinMaxScaler().fit_transform(time_features)
    df['time_vector'] = list(time_scaled)

    def lag_matrix(x, lag):
        n = len(x)
        X = np.zeros((n, lag))
        for i in range(1, lag + 1):
            X[i:, i - 1] = x[:-i]
        return X[:, ::-1]

    df[f'{target_col}_vector'] = list(lag_matrix(values, target_lag))

    arrs = {col: df[col].to_numpy(float) for col in features}
    for col, arr in tqdm(arrs.items(), desc="Lagging features"):
        df[f'{col}_vector'] = list(lag_matrix(arr, lag_value))

    return df


def prepare_X_y_ml(df, time_col, target_col, time_lag, horizon, val_frac=0.1, test_frac=0.1):
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col, ascending=True).reset_index(drop=True)
    values_cols = [c for c in df.columns if c not in [time_col, target_col]]
    target_vec = f"{target_col}_vector"
    print(df.columns)
    time_mat = np.stack(df["time_vector"].values)
    target_mat = np.stack(df[target_vec].values)
    values_mats = [np.stack(df[c].values) for c in values_cols]
    values_mats = [v.reshape(len(v), -1) for v in values_mats]
    feature_mat = np.concatenate(values_mats, axis=1)

    combined_mat = np.concatenate([time_mat, feature_mat, target_mat], axis=1)
    len_df = len(df)

    X_list, y_list = [], []
    for idx in tqdm(range(time_lag - 1, len_df - horizon), desc="Building X_y"):
        X_window = combined_mat[idx - time_lag + 1 : idx + 1]
        X_list.append(X_window)
        y_window = df[target_col].values[idx + 1 : idx + 1 + horizon]
        if len(y_window) < horizon:
            y_window = np.pad(y_window, (0, horizon - len(y_window)), constant_values=0.0)
        y_list.append(y_window)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    n_samples = len(X)
    n_test = int(n_samples * test_frac)
    n_val = int(n_samples * val_frac)
    n_train = n_samples - n_val - n_test

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[-n_test:], y[-n_test:]

    return X_train, y_train, X_val, y_val, X_test, y_test



def prepare_X_y_reg(df, time_col, time_cols, target_col, lag, val_frac=0.1, test_frac=0.1):
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col, ascending=True).reset_index(drop=True)
    df = df.copy()
    for i in range(1, lag + 1):
        df[f"{target_col}_lag_{i}"] = df[target_col].shift(i)
    df = df.dropna().reset_index(drop=True)

    X = df[time_cols + [f"{target_col}_lag_{i}" for i in range(lag, 0, -1)]].values
    y = df[target_col].values

    n_samples = len(X)
    n_test = int(n_samples * test_frac)
    n_val = int(n_samples * val_frac)
    n_train = n_samples - n_val - n_test

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[-n_test:], y[-n_test:]

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    import os

    home_path = os.getcwd()
    example_data_path = preprocess_data_path = os.path.join(home_path, "src", "preprocess_data", "Мойка.csv")
    df_wash = pd.read_csv(example_data_path)

    time_col = "По дням"
    target_col = "Количество заказ-нарядов"

    df_wash[time_col] = pd.to_datetime(df_wash[time_col], errors='coerce')
    df_wash['year'] = df_wash[time_col].dt.year
    df_wash['month'] = df_wash[time_col].dt.month
    df_wash['day'] = df_wash[time_col].dt.day

    time_cols = ["year", "month", "day"]

    lag_value = 3
    target_lag = 7
    time_lag = 2
    horizon = 360

    df = build_vectors(df=df_wash, time_col=time_col, time_cols=time_cols, target_col=target_col, lag_value=lag_value, target_lag=target_lag)
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_X_y_ml(df=df, time_col=time_col, target_col=target_col, time_lag=time_lag, horizon=horizon)

    print(df_wash.head())

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_X_y_reg(df=df_wash, time_col=time_col, time_cols=time_cols, target_col=target_col, lag=target_lag)

    print(X_reg)

