import os
import pandas as pd
from collections import defaultdict

home_path = os.getcwd()
init_data_path = os.path.join(home_path, "src", "init_data")
files = [f for f in os.listdir(init_data_path) if os.path.isfile(os.path.join(init_data_path, f))]
preprocess_data_path = os.path.join(home_path, "src", "preprocess_data")

data_by_years = []

def restore_timeseries(df, time_col, freq):
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    full_range = pd.date_range(start=df[time_col].min(), end=df[time_col].max(), freq=freq)
    df_full = pd.DataFrame({time_col: full_range})
    df = pd.merge(df_full, df, on=time_col, how='left')
    return df

for file in files:

    df_year = pd.read_excel(os.path.join(init_data_path, file))

    search_col_name_line = "По дням"
    lines_to_drop = ["Ижевск Трак Сервис ООО", "Итого"]
    lines_to_detect_first_line = ["Мойка", "Шиномонтаж", "Автосервис ИТС", "Автосервис"]
    lines_to_separate_data = ["Мойка", "Шиномонтаж", "Автосервис ИТС", "Автосервис", "ИП"]

    row = df_year[df_year.apply(lambda r: r.isin([search_col_name_line]).any(), axis=1)]
    if not row.empty:
        cols = row.iloc[0].tolist()


    mask_drop = df_year.apply(lambda row: row.isin(lines_to_drop).any(), axis=1)
    df_year = df_year[~mask_drop]

    mask_search = df_year.apply(lambda row: row.isin(lines_to_detect_first_line).any(), axis=1)
    indices = df_year.index[mask_search]
    min_index = indices.min() if not indices.empty else None

    df_year = df_year[min_index-1:].reset_index(drop=True)
    df_year.columns = cols

    result = []

    for idx, val in df_year["По дням"].items():
        if isinstance(val, str) and any(c.isalpha() for c in val):
            result.append([val, idx])

    data_indexes = []

    last_index = len(df_year)

    for i in range(len(result)):
        data = result[i]
        if i < len(result) - 1:
            next_data = result[i+1]
            data.append(next_data[1])
        else:
            data.append(last_index)

        data_indexes.append(data)


    dfs_dict = {}

    for data in data_indexes:
        data_name = data[0]
        data_start_index = data[1] + 1
        data_end_index = data[2]
        df_data = df_year.iloc[data_start_index: data_end_index]
        df_data = df_data.loc[:, ~df_data.columns.isna()]

        dfs_dict[data_name] = df_data

    data_by_years.append(dfs_dict)


result = defaultdict(list)

for d in data_by_years:
    for k, v in d.items():
        result[k].append(v)

result = {k: pd.concat(v, ignore_index=True) for k, v in result.items()}

time_col = "По дням"

for k, df in result.items():
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
        for col in df.columns:
            if col != time_col:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.astype({col: float for col in df.columns if col != time_col})
        df = df.sort_values(by=time_col).reset_index(drop=True)
        df[time_col] = df[time_col].dt.strftime("%Y-%m-%d %H:%M:%S")
        # df = df.dropna(axis=1, how="all")
        df = restore_timeseries(df, "По дням", "1D")

        result[k] = df


for k, df in result.items():
    path = os.path.join(preprocess_data_path, f"{k}.csv")
    df.to_csv(path, index=False)

