import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

#
# def prepare_X_y_ml(df, time_col, target_col, time_lag, horizon, val_frac=0.1, test_frac=0.1):
#     df[time_col] = pd.to_datetime(df[time_col])
#     df = df.sort_values(time_col, ascending=True).reset_index(drop=True)
#     values_cols = [c for c in df.columns if c not in [time_col, target_col]]
#     target_vec = f"{target_col}_vector"
#     print(df.columns)
#     time_mat = np.stack(df["time_vector"].values)
#     target_mat = np.stack(df[target_vec].values)
#     values_mats = [np.stack(df[c].values) for c in values_cols]
#     values_mats = [v.reshape(len(v), -1) for v in values_mats]
#     feature_mat = np.concatenate(values_mats, axis=1)
#
#     combined_mat = np.concatenate([time_mat, feature_mat, target_mat], axis=1)
#     len_df = len(df)
#
#     X_list, y_list = [], []
#     for idx in tqdm(range(time_lag - 1, len_df - horizon), desc="Building X_y"):
#         X_window = combined_mat[idx - time_lag + 1 : idx + 1]
#         X_list.append(X_window)
#         y_window = df[target_col].values[idx + 1 : idx + 1 + horizon]
#         if len(y_window) < horizon:
#             y_window = np.pad(y_window, (0, horizon - len(y_window)), constant_values=0.0)
#         y_list.append(y_window)
#
#     X = np.array(X_list, dtype=np.float32)
#     y = np.array(y_list, dtype=np.float32)
#
#     n_samples = len(X)
#     n_test = int(n_samples * test_frac)
#     n_val = int(n_samples * val_frac)
#     n_train = n_samples - n_val - n_test
#
#     X_train, y_train = X[:n_train], y[:n_train]
#     X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
#     X_test, y_test = X[-n_test:], y[-n_test:]
#
#     return X_train, y_train, X_val, y_val, X_test, y_test

def prepare_df_ml(df, time_col, target_col, time_lag, horizon, val_frac=0.1, test_frac=0.1):
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col, ascending=True).reset_index(drop=True)
    values_cols = [c for c in df.columns if c not in [time_col, target_col]]
    target_vec = f"{target_col}_vector"
    time_mat = np.stack(df["time_vector"].values)
    target_mat = np.stack(df[target_vec].values)
    values_mats = [np.stack(df[c].values) for c in values_cols]
    values_mats = [v.reshape(len(v), -1) for v in values_mats]
    feature_mat = np.concatenate(values_mats, axis=1)
    combined_mat = np.concatenate([time_mat, feature_mat, target_mat], axis=1)
    len_df = len(df)
    data = []
    for idx in range(time_lag - 1, len_df - horizon):
        X_window = combined_mat[idx - time_lag + 1: idx + 1]
        y_window = df[target_col].values[idx + 1: idx + 1 + horizon]
        if len(y_window) < horizon:
            y_window = np.pad(y_window, (0, horizon - len(y_window)), constant_values=0.0)
        data.append({"X": X_window, "y": y_window})
    n_samples = len(data)
    n_test = int(n_samples * test_frac)
    n_val = int(n_samples * val_frac)
    n_train = n_samples - n_val - n_test
    df_train = pd.DataFrame(data[:n_train])
    df_val = pd.DataFrame(data[n_train:n_train + n_val])
    df_test = pd.DataFrame(data[-n_test:])
    return df_train, df_val, df_test


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


# def prepare_X_y_reg_df(df, time_col, time_cols, target_col, lag):
#     df[time_col] = pd.to_datetime(df[time_col])
#     df = df.sort_values(time_col).reset_index(drop=True).copy()
#     for i in range(1, lag + 1):
#         df[f"{target_col}_lag_{i}"] = df[target_col].shift(i)
#     df = df.dropna().reset_index(drop=True)
#     cols = time_cols + [f"{target_col}_lag_{i}" for i in range(lag, 0, -1)] + [target_col]
#     return df[cols]


def prepare_X_y_reg_df(df, time_col, time_cols, target_col, lag):
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True).copy()

    df['year'] = df[time_col].dt.year
    df['month'] = df[time_col].dt.month
    df['day'] = df[time_col].dt.day
    df['dow'] = df[time_col].dt.dayofweek

    df['year_sin'] = np.sin(2 * np.pi * (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min() + 1))
    df['year_cos'] = np.cos(2 * np.pi * (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min() + 1))
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    for i in range(1, lag + 1):
        df[f"{target_col}_lag_{i}"] = df[target_col].shift(i)

    df = df.dropna().reset_index(drop=True)

    lag_cols = [f"{target_col}_lag_{i}" for i in range(lag, 0, -1)]
    time_feat_cols = ['year', 'month', 'day', 'year_sin', 'year_cos', 'month_sin', 'month_cos', 'dow', 'dow_sin', 'dow_cos', 'day_sin', 'day_cos']
    cols = time_cols + time_feat_cols + lag_cols + [target_col]

    return df[cols]


def recursive_forecast(model, df, time_col, time_cols, target_col, lag, horizon):
    df = df.sort_values(time_col).copy().reset_index(drop=True)
    df = df.copy()

    preds = []
    df_forecast = df.copy()

    for _ in range(horizon):
        # создаём временные признаки для последней строки
        row = df_forecast.iloc[[-1]].copy()
        date = row[time_col].values[0]
        year = pd.Timestamp(date).year
        month = pd.Timestamp(date).month
        day = pd.Timestamp(date).day
        dow = pd.Timestamp(date).dayofweek

        row['year'] = year
        row['month'] = month
        row['day'] = day
        row['dow'] = dow
        row['year_sin'] = np.sin(2 * np.pi * (year - df[time_col].dt.year.min()) / (df[time_col].dt.year.max() - df[time_col].dt.year.min() + 1))
        row['year_cos'] = np.cos(2 * np.pi * (year - df[time_col].dt.year.min()) / (df[time_col].dt.year.max() - df[time_col].dt.year.min() + 1))
        row['month_sin'] = np.sin(2 * np.pi * month / 12)
        row['month_cos'] = np.cos(2 * np.pi * month / 12)
        row['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        row['dow_cos'] = np.cos(2 * np.pi * dow / 7)
        row['day_sin'] = np.sin(2 * np.pi * day / 31)
        row['day_cos'] = np.cos(2 * np.pi * day / 31)

        # создаём lag признаки
        for i in range(1, lag + 1):
            row[f"{target_col}_lag_{i}"] = df_forecast[target_col].iloc[-i]

        lag_cols = [f"{target_col}_lag_{i}" for i in range(lag, 0, -1)]
        time_feat_cols = ['year', 'month', 'day', 'year_sin', 'year_cos', 'month_sin', 'month_cos',
                          'dow', 'dow_sin', 'dow_cos', 'day_sin', 'day_cos']

        X_row = row[time_feat_cols + lag_cols].values.reshape(1, -1)
        pred = model.predict(X_row)[0]

        preds.append(pred)

        # создаём новую строку для рекурсии
        next_date = pd.Timestamp(date) + pd.Timedelta(days=1)
        new_row = pd.DataFrame({time_col: [next_date], target_col: [pred]})
        df_forecast = pd.concat([df_forecast, new_row], ignore_index=True)

    return preds


if __name__ == "__main__":
    import os

    home_path = os.getcwd()
    example_data_path = preprocess_data_path = os.path.join(home_path, "src", "preprocess_data", "features", "Мойка_features.csv")
    x_y_data_path = os.path.join(home_path, "src", "preprocess_data", "X_y")
    if not os.path.exists(x_y_data_path):
        os.makedirs(x_y_data_path)

    df_wash = pd.read_csv(example_data_path)

    time_col = "date"
    target_col = "Количество заказ-нарядов"

    # time_cols = ["month", "month_sin", "month_cos", "dow", "dow_sin", "dow_cos",]
    col_to_one_hot = ["month", "dow"]
    col_to_drop = ["is_friday", "is_monday"]
    df_wash = df_wash.drop(columns=col_to_drop)
    df_wash = df_wash.dropna()


    lag_value = 3
    target_lag = 2
    time_lag = 2
    horizon = 30*3
    time_cols = ["date",]

    # df = build_vectors(df=df_wash, time_col=time_col, time_cols=time_cols, target_col=target_col, lag_value=lag_value, target_lag=target_lag)
    # df = pd.get_dummies(df, columns=col_to_one_hot, drop_first=False)
    # # df_train, df_val, df_test = prepare_df_ml(df=df, time_col=time_col, target_col=target_col, time_lag=time_lag, horizon=horizon)

    df = prepare_X_y_reg_df(df=df_wash, time_col=time_col, time_cols=time_cols, target_col=target_col, lag=target_lag)

    df_train = df[:len(df) - 30*3]
    df_start = df_train[len(df_train) - target_lag*2:]
    df_start = df_start[["date", "Количество заказ-нарядов"]]

    df_train = df_train.drop(columns=["date"])

    print(f"df_train cols = {df_train.columns}")

    df_test = df[len(df) - horizon:]
    y_true = df_test["Количество заказ-нарядов"].to_list()

    X = df_train.drop(columns=["Количество заказ-нарядов"])
    y = df_train["Количество заказ-нарядов"]

    model = XGBRegressor(n_estimators=15000, max_depth=16, learning_rate=0.1, verbose=1)
    model.fit(X, y)

    y_pred = recursive_forecast(model, df, time_col, time_cols, target_col, target_lag, horizon)
    print(y_pred)
    print(y_true)
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R2:", r2_score(y_true, y_pred))
    print("MAPE:", np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100)

    # print(df_test.head)
