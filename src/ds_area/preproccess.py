import os
import pandas as pd
from collections import defaultdict
import requests
from datetime import datetime, timezone
from typing import List, Dict


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

# Преобразуем колонку в тип datetime
df[time_col] = pd.to_datetime(df[time_col])

# Разделяем на компоненты
df['year'] = df[time_col].dt.year
df['month'] = df[time_col].dt.month
df['day'] = df[time_col].dt.day

#Добавление данных по погоде
def fetch_realmeteo_year(city_slug: str, station_num: int, year: int, *,
                         session: requests.Session = None,
                         headers: Dict[str, str] = None) -> dict:
    """Скачивает JSON архива за год с realmeteo.ru"""
    url = f"https://realmeteo.ru/{city_slug}/{station_num}/history/{year}.json"
    _headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/122 Safari/537.36"),
        "Referer": f"https://realmeteo.ru/{city_slug}/{station_num}/history",
        "Accept": "application/json,text/plain,*/*",
    }
    if headers:
        _headers.update(headers)
    sess = session or requests.Session()
    r = sess.get(url, headers=_headers, timeout=30)
    r.raise_for_status()
    return r.json()

def history_json_to_df(j: dict, tz_local: str = "Europe/Samara") -> pd.DataFrame:
    """
    Превращает годовой JSON в DataFrame c 30-минутной частотой.
    Колонки равны ключам из j['data'].
    Добавляет datetime(локальный) и date(локальный).
    """
    # В JSON: stop (секунды), interval (мс), data: {series: [..]}
    stop_s = int(j["stop"])
    interval_ms = int(j["interval"])
    step = pd.to_timedelta(interval_ms, unit="ms")
    # длину берём по любому доступному ряду
    any_key = next(k for k, v in j["data"].items() if isinstance(v, list))
    n = len(j["data"][any_key])

    # start = stop - (n-1)*interval, шкала в UTC
    start_utc = pd.to_datetime(stop_s, unit="s", utc=True) - step * (n - 1)
    dt_utc = pd.date_range(start=start_utc, periods=n, freq=step, tz="UTC")

    # в локальную зону метеостанции (Ижевск фактически +4)
    dt_local = dt_utc.tz_convert(tz_local)

    df = pd.DataFrame({"datetime": dt_local})
    # перенесём ряды
    for k, arr in j["data"].items():
        if isinstance(arr, list) and len(arr) == n:
            df[k] = arr

    df["date"] = df["datetime"].dt.date  # локальная дата
    return df

def get_realmeteo_daily(city_slug: str = "izhevsk",
                        station_num: int = 1,
                        years: List[int] = list(range(2020, 2026)),
                        tz_local: str = "Europe/Samara") -> pd.DataFrame:
    """
    Скачивает годы, конкатенирует 30-минутные данные, агрегирует по дням в фичи.
    Возвращает датафрейм с колонкой date и дневными признаками.
    """
    sess = requests.Session()
    parts = []
    for y in years:
        j = fetch_realmeteo_year(city_slug, station_num, y, session=sess)
        parts.append(history_json_to_df(j, tz_local=tz_local))
    halfhour = pd.concat(parts, ignore_index=True).sort_values("datetime")

    agg_map = {}
    if "temperature" in halfhour.columns:
        agg_map["temperature"] = "mean"          # среднесуточная t
    if "pressure" in halfhour.columns:
        agg_map["pressure"] = "mean"
    if "humidity" in halfhour.columns:
        agg_map["humidity"] = "mean"
    if "wind_speed_avg" in halfhour.columns:
        agg_map["wind_speed_avg"] = ["mean", "max"]
    if "wind_speed_hi" in halfhour.columns:
        agg_map["wind_speed_hi"] = "max"         # максимум порывов
    if "solar_rad" in halfhour.columns:
        agg_map["solar_rad"] = "sum"             # суммарная радиация за день
    if "uv" in halfhour.columns:
        agg_map["uv"] = "max"                    # дневной максимум
    if "pcp" in halfhour.columns:
        agg_map["pcp"] = "sum"                   # суммарные осадки за день (мм)

    if not agg_map:
        raise ValueError("В ответе нет распознанных погодных рядов.")

    daily = (
        halfhour
        .groupby("date", as_index=False)
        .agg(agg_map)
    )

    daily.columns = [
        "date" if c[0] == "date" else
        c[0] if isinstance(c, str) else
        (f"{c[0]}_{c[1]}")
        for c in daily.columns
    ]
    return daily


# Скачиваем и готовим погодные фичи:
daily_weather = get_realmeteo_daily(
    city_slug="izhevsk", station_num=1, years=list(range(2019, 2026)),
    tz_local="Europe/Samara")


# One‑hot энкодинг
def one_hot_encode(df: pd.DataFrame, cat_col: str) -> pd.DataFrame:
    """One‑hot (dummy) encode a categorical column"""
    return pd.get_dummies(df, columns=[cat_col], prefix=cat_col)

# Label энкодинг
def label_encode(df: pd.DataFrame, cat_col: str) -> pd.DataFrame:
    """Label-encode a categorical column into integer codes, keeping the original."""
    df_copy = df.copy()
    df_copy[f"{cat_col}_encoded"], uniques = pd.factorize(df_copy[cat_col], sort=True)
    return df_copy

# Пример вызова функции
# df = one_hot_encode(df, 'year') # добавляет колонки year_2020, year_2021, year_2022	и т.д.
# df = label_encode(df, 'year') # добавляет колонку year_encoded

for k, df in result.items():
    to_drop_cols = ["wind_speed_hi_max"]
    path = os.path.join(preprocess_data_path, f"{k}.csv")
    df['date'] = pd.to_datetime(df['По дням']).dt.date
    df = df.merge(daily_weather, on="date", how="left")
    df = df.drop(columns=to_drop_cols)
    df.to_csv(path, index=False)

