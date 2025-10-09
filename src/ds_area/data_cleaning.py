"""
Data Cleaning and Preprocessing for Time Series with GAN-based Imputation

Цель: Очистка данных, удаление дубликатов, обработка выбросов и заполнение пропусков с помощью GAN.
Акцент: Использование GAN (TimeGAN) для восстановления пропущенных значений в временных рядах.
Поддержка многомерных временных рядов (все числовые колонки).
Визуализация результатов в HTML (для VS Code).
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
import warnings
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# Путь относительно расположения скрипта
SCRIPT_DIR = Path(__file__).parent.parent  # поднимаемся из ds_area/ в src/
DATA_DIR = SCRIPT_DIR / "preprocess_data"
OUTPUT_DIR = DATA_DIR / "cleaned"
REPORTS_DIR = OUTPUT_DIR / "reports"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Создаём папки для вывода
for d in [OUTPUT_DIR, REPORTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Поиск CSV-файлов
if DATA_DIR.exists():
    FILES = [f.name for f in DATA_DIR.glob("*.csv")]
    print("Найденные CSV файлы для обработки:")
    for f in FILES:
        print(f" - {f}")
else:
    print(f"Папка не найдена: {DATA_DIR}")
    FILES = []

# Проверка наличия synthcity
try:
    from synthcity.plugins.imputation import ImputationPlugin
    GAN_AVAILABLE = True
except ImportError:
    print("  Библиотека synthcity не установлена. Будет использована интерполяция.")
    GAN_AVAILABLE = False


def detect_time_and_numeric_columns(df):
    """Автоматическое определение временной и числовых колонок."""
    time_col = None
    numeric_cols = []

    for col in df.columns:
        col_lower = col.lower()
        if 'день' in col_lower or 'время' in col_lower or 'date' in col_lower or 'datetime' in col_lower:
            time_col = col
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    # Если не нашли время — пробуем первую колонку
    if time_col is None:
        time_col = df.columns[0]
        print(f"ℹ  Временная колонка не распознана, используется: {time_col}")

    # Если числовых колонок нет — попробуем все, кроме времени
    if not numeric_cols:
        numeric_cols = [c for c in df.columns if c != time_col]

    return time_col, numeric_cols


def clean_and_impute_multivariate(df, time_col, numeric_cols):
    """
    Очистка и импутация для многомерного временного ряда.
    Возвращает (очищенный DataFrame, отчёт).
    """
    report = []
    original_shape = df.shape

    # 1. Удаление дубликатов по времени
    df = df.drop_duplicates(subset=[time_col], keep='first').reset_index(drop=True)
    dup_removed = original_shape[0] - df.shape[0]
    if dup_removed > 0:
        report.append(f"- Удалено дубликатов по времени: {dup_removed}")

    # 2. Преобразование времени
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    invalid_time = df[time_col].isna().sum()
    if invalid_time > 0:
        df = df.dropna(subset=[time_col]).reset_index(drop=True)
        report.append(f"- Удалено строк с некорректным временем: {invalid_time}")

    # 3. Обработка выбросов (осторожно!)
    outlier_info = {}
    for col in numeric_cols:
        if col in df.columns:
            valid_data = df[col].dropna()
            if len(valid_data) > 20:  # достаточно данных
                clf = IsolationForest(contamination=0.03, random_state=42)
                preds = clf.fit_predict(valid_data.values.reshape(-1, 1))
                n_outliers = (preds == -1).sum()
                if n_outliers > 0:
                    outlier_info[col] = n_outliers
                    # Не удаляем, а помечаем — но для импутации лучше временно удалить аномалии?
                    # Здесь оставим, но в отчёте укажем
            else:
                outlier_info[col] = "недостаточно данных"

    if outlier_info:
        report.append("- Обнаружены потенциальные выбросы (не удалены, только зафиксированы):")
        for col, cnt in outlier_info.items():
            report.append(f"    {col}: {cnt}")

    # 4. Подготовка данных для импутации
    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
    df_indexed = df_sorted.set_index(time_col)

    # Убедимся, что все numeric_cols числовые
    for col in numeric_cols:
        df_indexed[col] = pd.to_numeric(df_indexed[col], errors='coerce')

    # Подсчёт пропусков до
    missing_before = df_indexed[numeric_cols].isna().sum()

    # 5. Импутация
    # Подсчёт пропусков до
    missing_ratio = df_indexed[numeric_cols].isna().sum().sum() / (len(df_indexed) * len(numeric_cols))

    imputation_method = "нет пропусков"
    if df_indexed[numeric_cols].isna().sum().sum() > 0:
        if GAN_AVAILABLE and missing_ratio < 0.4:  # Если пропусков < 40%
            try:
                df_freq = df_indexed.asfreq('D')  # TimeGAN требует частоты
                plugin = ImputationPlugin("timegan", n_iter=100, batch_size=64, device="cpu")
                imputed = plugin.fit_transform(df_freq[numeric_cols])
                imputed_original_dates = imputed.loc[df_indexed.index]
                df_indexed[numeric_cols] = imputed_original_dates
                imputation_method = "GAN (TimeGAN)"
            except Exception as e:
                imputation_method = f"GAN не сработал: {str(e)[:60]}... → переключаемся на интерполяцию"
                # Переходим к fallback
                for col in numeric_cols:
                    df_indexed[col] = df_indexed[col].interpolate(method='linear', limit_direction='both')
                    if df_indexed[col].isna().any():
                        df_indexed[col] = df_indexed[col].fillna(method='ffill').fillna(method='bfill')
        else:
            # Fallback: интерполяция + бизнес-логика
            for col in numeric_cols:
                df_indexed[col] = df_indexed[col].interpolate(method='linear', limit_direction='both')
                if df_indexed[col].isna().any():
                    df_indexed[col] = df_indexed[col].fillna(method='ffill').fillna(method='bfill')

            imputation_method = "Интерполяция + FFill/BFill"

        # === Бизнес-логика применяется ВСЕГДА ===
        if "Запчастей" in numeric_cols and "Всего по заказ-наряду" in df_indexed.columns and "Работ" in df_indexed.columns:
            computed = df_indexed["Всего по заказ-наряду"] - df_indexed["Работ"]
            mask = computed.notna() & (computed >= 0)
            df_indexed.loc[mask, "Запчастей"] = df_indexed.loc[mask, "Запчастей"].fillna(computed[mask])

        # Заполняем остатки нулями
        for col in numeric_cols:
            if df_indexed[col].isna().any():
                df_indexed[col] = df_indexed[col].fillna(0)

        report.append(f"- Заполнение пропусков: {imputation_method}")
        missing_after = df_indexed[numeric_cols].isna().sum()
        report.append(f"- Пропусков до: {missing_before.to_dict()}")
        report.append(f"- Пропусков после: {missing_after.to_dict()}")
    else:
        report.append("- Пропусков не обнаружено")

    
    # Возврат к DataFrame с колонкой времени
    df_final = df_indexed.reset_index()

    return df_final, report


def plot_comparison_html(original_df, cleaned_df, time_col, numeric_cols, filename):
    """Создаёт HTML-визуализацию до/после и сохраняет в файл."""
    n_cols = len(numeric_cols)
    if n_cols == 0:
        return

    fig = make_subplots(
        rows=n_cols, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=numeric_cols
    )

    for i, col in enumerate(numeric_cols, start=1):
        # Исходные данные (только где не NaN)
        orig = original_df.dropna(subset=[col])
        clean = cleaned_df  # уже без NaN

        fig.add_trace(
            go.Scatter(
                x=orig[time_col], y=orig[col],
                mode='markers', name=f'{col} (исходный)',
                marker=dict(size=6, color='lightcoral')
            ),
            row=i, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=clean[time_col], y=clean[col],
                mode='lines+markers', name=f'{col} (очищенный)',
                line=dict(width=2),
                marker=dict(size=4, color='steelblue')
            ),
            row=i, col=1
        )

    fig.update_layout(
        height=300 * n_cols,
        title_text=f"Сравнение до и после очистки: {filename}",
        showlegend=False,
        hovermode='x unified'
    )

    plot_path = PLOTS_DIR / f"plot_{filename}.html"
    fig.write_html(plot_path)
    return plot_path


def process_file(filepath, filename):
    """Обработка одного файла."""
    print(f"\n Обработка файла: {filename}")
    df_orig = pd.read_csv(filepath, encoding='utf-8')

    time_col, numeric_cols = detect_time_and_numeric_columns(df_orig)
    print(f"  Временная колонка: {time_col}")
    print(f"  Числовые колонки: {numeric_cols}")

    df_cleaned, report = clean_and_impute_multivariate(df_orig.copy(), time_col, numeric_cols)

    # Визуализация
    plot_path = plot_comparison_html(df_orig, df_cleaned, time_col, numeric_cols, filename)

    # Сохранение
    output_path = OUTPUT_DIR / f"cleaned_{filename}"
    df_cleaned.to_csv(output_path, index=False, encoding='utf-8')

    # Сохранение отчёта
    report_path = REPORTS_DIR / f"report_{filename}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Отчёт по файлу: {filename}\n")
        f.write("="*50 + "\n")
        f.write("\n".join(report))
        f.write(f"\n\nВизуализация: {plot_path.relative_to(SCRIPT_DIR)}")

    print(f"  Сохранено: {output_path.name}")
    print(f"  Отчёт: {report_path.name}")
    print(f"  График: {plot_path.name}")

    return df_cleaned, report


if __name__ == "__main__":
    print(" Начало очистки данных с использованием GAN (TimeGAN)...")
    print(f"GAN доступен: {'Да' if GAN_AVAILABLE else 'Нет (используем интерполяцию)'}")

    all_reports = {}
    for file in FILES:
        filepath = DATA_DIR / file
        if filepath.exists():
            df_cleaned, report = process_file(filepath, file)
            all_reports[file] = report
        else:
            print(f"  Файл не найден: {file}")


    print(f"\n Очистка завершена! Результаты в папке: {OUTPUT_DIR.relative_to(SCRIPT_DIR)}")