import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import os
from pathlib import Path
from datetime import datetime

# Путь относительно расположения скрипта
SCRIPT_DIR = Path(__file__).parent.parent  # поднимаемся из ds_area/ в src/
DATA_DIR = SCRIPT_DIR / "preprocess_data"
INPUT_DIR = DATA_DIR / "cleaned"
OUTPUT_DIR = DATA_DIR / "features"

# Убедимся, что выходная папка существует
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_features(df, target_col='Всего по заказ-наряду', 
                      work_col='Работ', parts_col='Запчастей', 
                      orders_col='Количество заказ-нарядов',
                      peak_months=None, has_parts=True,
                      freq='D', seasonal_period=7):
    """
    Генерация признаков для временного ряда.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['По дням'])
    df = df.set_index('date').sort_index()
    
    # Заполним пропуски нулями (дни без активности)
    full_date_range = pd.date_range(df.index.min(), df.index.max(), freq='D')
    df = df.reindex(full_date_range).fillna(0)
    df.index.name = 'date'
    
    # === 1. Календарные признаки ===
    df['dow'] = df.index.dayofweek  # 0=Пн, 6=Вс
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['is_friday'] = (df['dow'] == 4).astype(int)
    df['is_monday'] = (df['dow'] == 0).astype(int)
    
    # Пиковые месяцы
    if peak_months is None:
        peak_months = list(range(1, 13))
    df['is_peak_month'] = df['month'].isin(peak_months).astype(int)
    
    # === 2. Лаги и скользящие средние ===
    for lag in [1, 7, 14, 28]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)        
        df[f'lag_{lag}_orders'] = df[orders_col].shift(lag)
    
    for window in [7, 14, 28]:
        df[f'ma{window}'] = df[target_col].rolling(window, min_periods=1).mean()
        df[f'std{window}'] = df[target_col].rolling(window, min_periods=1).std().fillna(0)
        df[f'growth_wow_{window}'] = df[target_col] - df[target_col].shift(7)
    
    # === 3. STL-разложение ===
    try:
        if len(df) >= 2 * seasonal_period:
            stl = STL(df[target_col], seasonal=seasonal_period, robust=True).fit()
            df['stl_trend'] = stl.trend
            df['stl_seasonal'] = stl.seasonal
            df['stl_resid'] = stl.resid
            resid_std = df['stl_resid'].std()
            if pd.notna(resid_std) and resid_std > 0:
                df['is_anomaly'] = (np.abs(df['stl_resid']) > 3 * resid_std).astype(int)
            else:
                df['is_anomaly'] = 0
        else:
            df['stl_trend'] = np.nan
            df['stl_seasonal'] = np.nan
            df['stl_resid'] = np.nan
            df['is_anomaly'] = 0
    except Exception as e:
        print(f"STL failed: {e}")
        df['stl_trend'] = np.nan
        df['stl_seasonal'] = np.nan
        df['stl_resid'] = np.nan
        df['is_anomaly'] = 0
    
    # === 4. Взаимодействия ===
    df['dow_x_month'] = df['dow'] * df['month']
    df['is_friday_dec'] = ((df['dow'] == 4) & (df['month'] == 12)).astype(int)
    df['is_friday_mar'] = ((df['dow'] == 4) & (df['month'] == 3)).astype(int)
    
    # === 5. Бизнес-метрики ===
    df['avg_check'] = df[target_col] / (df[orders_col] + 1e-6)
    if work_col in df.columns and not df[work_col].isna().all():
        df['work_share'] = df[work_col] / (df[target_col] + 1e-6)
    if has_parts and parts_col in df.columns and not df[parts_col].isna().all():
        df['parts_share'] = df[parts_col] / (df[target_col] + 1e-6)
    
    # Удаляем строки с пропусками в таргете (начало ряда)
    df = df[df[target_col].notna()]
    
    return df

# Список всех создаваемых признаков (для отчёта)
FEATURE_CATEGORIES = {
    "Календарные признаки": [
        "dow", "dow_sin", "dow_cos", "month", "month_sin", "month_cos",
        "is_weekend", "is_friday", "is_monday", "is_peak_month"
    ],
    "Лаги и скользящие средние": [
        "lag_1", "lag_7", "lag_14", "lag_28",
        "lag_1_orders", "lag_7_orders", "lag_14_orders", "lag_28_orders",
        "ma7", "ma14", "ma28", "std7", "std14", "std28", "growth_wow_7"
    ],
    "STL-компоненты": [
        "stl_trend", "stl_seasonal", "stl_resid", "is_anomaly"
    ],
    "Взаимодействия": [
        "dow_x_month", "is_friday_dec", "is_friday_mar"
    ],
    "Бизнес-метрики": [
        "avg_check", "work_share", "parts_share"
    ]
}

# Конфигурация датасетов
datasets = {
    "Автосервис ИТС": {
        "file": INPUT_DIR / "cleaned_Автосервис ИТС.csv",
        "peak_months": [2, 4, 8],
        "has_parts": True
    },
    "Мойка": {
        "file": INPUT_DIR / "cleaned_Мойка.csv",
        "peak_months": [3, 11, 12],
        "has_parts": False
    },
    "Шиномонтаж": {
        "file": INPUT_DIR / "cleaned_Шиномонтаж.csv",
        "peak_months": [10, 11, 4],
        "has_parts": False
    }
}

# Сбор информации для отчёта
report_lines = []
report_lines.append("=" * 60)
report_lines.append("ОТЧЁТ ПО ГЕНЕРАЦИИ ПРИЗНАКОВ (FEATURE ENGINEERING)")
report_lines.append("=" * 60)
report_lines.append(f"Дата и время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append(f"Папка с исходными данными: {INPUT_DIR}")
report_lines.append(f"Папка с результатами: {OUTPUT_DIR}")
report_lines.append("")

# Обработка каждого датасета
results = {}
errors = []

for name, config in datasets.items():
    print(f"Обработка: {name}")
    try:
        df_raw = pd.read_csv(config["file"], parse_dates=['По дням'])
        df_feat = generate_features(
            df_raw,
            peak_months=config["peak_months"],
            has_parts=config["has_parts"]
        )
        output_file = OUTPUT_DIR / f"{name}_features.csv"
        df_feat.to_csv(output_file, index=True)
        results[name] = {
            "rows": len(df_feat),
            "file": output_file,
            "columns": df_feat.columns.tolist()
        }
        print(f"  Сохранено: {output_file} | строк: {len(df_feat)}")
    except Exception as e:
        error_msg = f"Ошибка при обработке {name}: {e}"
        errors.append(error_msg)
        print(f"  {error_msg}")

# Формирование отчёта
report_lines.append(" УСПЕШНО ОБРАБОТАНЫЕ ДАТАСЕТЫ:")
for name, info in results.items():
    report_lines.append(f"  • {name}")
    report_lines.append(f"    - Сохранён в: {info['file']}")
    report_lines.append(f"    - Количество строк: {info['rows']}")
    report_lines.append(f"    - Количество признаков: {len(info['columns'])}")

if errors:
    report_lines.append("\n ОШИБКИ:")
    for err in errors:
        report_lines.append(f"  • {err}")

report_lines.append("\n СПИСОК СОЗДАННЫХ ПРИЗНАКОВ ПО КАТЕГОРИЯМ:")
for category, features in FEATURE_CATEGORIES.items():
    report_lines.append(f"\n{category}:")
    for feat in features:
        report_lines.append(f"  - {feat}")

report_lines.append("\n" + "=" * 60)
report_lines.append("Готово! Признаки сгенерированы.")
report_lines.append("=" * 60)

# Сохранение отчёта
report_path = OUTPUT_DIR / "feature_engineering_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"\n Отчёт сохранён: {report_path}")