# forecast_30d.py
"""
Прогноз на 30 дней вперёд + автоматическая оценка качества на hold-out (последние 30 дней).
"""

import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from datetime import timedelta


SRC_DIR = Path(__file__).parent.parent
FEATURES_DIR = SRC_DIR / "preprocess_data" / "features"
MODELS_DIR = SRC_DIR / "models" / "saved"
FORECAST_DIR = SRC_DIR / "models" / "forecasts"
REPORT_DIR = SRC_DIR / "models" / "reports"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FORECAST_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

TARGET = "Всего по заказ-наряду"

BUSINESSES = {
    "Автосервис ИТС": {"peak_months": [2, 4, 8], "has_parts": True},
    "Мойка": {"peak_months": [3, 11, 12], "has_parts": False},
    "Шиномонтаж": {"peak_months": [4, 10, 11], "has_parts": False}
}

SHORT_FEATURES_BASE = [
    "lag_1", "lag_7", "ma7", "dow_sin", "dow_cos",
    "is_friday", "is_monday", "is_weekend",
    "smoothed_trend", "avg_check", "is_anomaly",
    "is_peak_month", "month_sin", "month_cos"
]

LONG_FEATURES = [
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_peak_month", "is_friday_dec", "is_friday_mar"
]

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8)) * 100

def calculate_metrics(y_true, y_pred, y_naive):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_val = mape(y_true, y_pred)
    smape_val = smape(y_true, y_pred)
    mae_model = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(y_true - y_naive))
    mase = mae_model / (mae_naive + 1e-8)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape_val,
        "sMAPE (%)": smape_val,
        "MASE": mase
    }

def create_future_features(start_date, n_days, peak_months):
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    df = pd.DataFrame(index=dates)
    df.index.name = 'date'
    df['dow'] = df.index.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['is_friday'] = (df['dow'] == 4).astype(int)
    df['is_monday'] = (df['dow'] == 0).astype(int)
    df['is_peak_month'] = df['month'].isin(peak_months).astype(int)
    df['is_friday_dec'] = ((df['dow'] == 4) & (df['month'] == 12)).astype(int)
    df['is_friday_mar'] = ((df['dow'] == 4) & (df['month'] == 3)).astype(int)
    return df

def train_and_save_models(df, business_name, peak_months):
    print(f"\nОбучение моделей для: {business_name}")
    df = df.copy()
    df['smoothed_trend'] = df[TARGET].rolling(window=14, min_periods=1).mean().ffill().bfill()
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['is_peak_month'] = df['month'].isin(peak_months).astype(int)
    
    df_train_short = df[df['is_anomaly'] == 0].copy()
    X_short = df_train_short[SHORT_FEATURES_BASE].fillna(0)
    y_short = df_train_short[TARGET]
    
    model_short = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        objective='reg:squarederror'
    )
    model_short.fit(X_short, y_short)

    feature_importance = pd.Series(model_short.feature_importances_, index=SHORT_FEATURES_BASE)
    plt.figure(figsize=(8, 6))
    feature_importance.sort_values().plot(kind='barh', title=f'Важность признаков: {business_name}')
    plt.xlabel('Важность (XGBoost)')
    plt.tight_layout()
    plt.savefig(MODELS_DIR / f"feature_importance_{business_name}.png")
    plt.close()
    print(f"   Сохранена важность признаков: feature_importance_{business_name}.png")

    X_long = df[LONG_FEATURES].fillna(0)
    y_long = df['smoothed_trend']
    trend_model = LinearRegression()
    trend_model.fit(X_long, y_long)
    
    joblib.dump(model_short, MODELS_DIR / f"{business_name}_short_xgb.pkl")
    joblib.dump(trend_model, MODELS_DIR / f"{business_name}_trend_lr.pkl")
    
    metadata = {
        "peak_months": peak_months,
        "short_features": SHORT_FEATURES_BASE,
        "long_features": LONG_FEATURES,
        "last_date": df.index[-1]
    }
    joblib.dump(metadata, MODELS_DIR / f"{business_name}_metadata.pkl")
    
    print(f" Модели сохранены для {business_name}")

def forecast_30_days(business_name):
    df = pd.read_csv(FEATURES_DIR / f"{business_name}_features.csv", parse_dates=["date"])
    df = df.set_index("date").sort_index()
    
    model_short = joblib.load(MODELS_DIR / f"{business_name}_short_xgb.pkl")
    trend_model = joblib.load(MODELS_DIR / f"{business_name}_trend_lr.pkl")
    metadata = joblib.load(MODELS_DIR / f"{business_name}_metadata.pkl")
    peak_months = metadata["peak_months"]
    
    last_date = df.index[-1]
    future_start = last_date + timedelta(days=1)
    future_df = create_future_features(future_start, 30, peak_months)
    
    forecast = []
    current_df = df.copy()
    
    for i in range(30):
        current_date = future_start + timedelta(days=i)
        dow = current_date.dayofweek
        month = current_date.month
        
        if i == 0:
            lag_1 = current_df[TARGET].iloc[-1]
            lag_7 = current_df[TARGET].iloc[-7] if len(current_df) >= 7 else lag_1
            ma7 = current_df[TARGET].tail(7).mean()
            avg_check = current_df['avg_check'].iloc[-1] if 'avg_check' in current_df.columns else 1.0
        else:
            lag_1 = forecast[-1]
            lag_7 = forecast[-7] if len(forecast) >= 7 else lag_1
            ma7 = np.mean(forecast[-7:]) if len(forecast) >= 7 else lag_1
            avg_check = current_df['avg_check'].mean() if 'avg_check' in current_df.columns else 1.0
        
        X_trend_df = future_df.loc[[current_date], metadata["long_features"]]
        smoothed_trend_pred = trend_model.predict(X_trend_df)[0]
        
        X_short = np.array([
            lag_1,
            lag_7,
            ma7,
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            1 if dow == 4 else 0,
            1 if dow == 0 else 0,
            1 if dow >= 5 else 0,
            smoothed_trend_pred,
            avg_check,
            0,
            1 if month in peak_months else 0,
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12)
        ]).reshape(1, -1)
        
        pred = model_short.predict(X_short)[0]
        forecast.append(max(0, pred))
    
    forecast_series = pd.Series(forecast, index=pd.date_range(future_start, periods=30, freq='D'))
    forecast_df = pd.DataFrame({"forecast": forecast_series})
    forecast_path = FORECAST_DIR / f"{business_name}_30d_forecast.csv"
    forecast_df.to_csv(forecast_path)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index[-60:], y=df[TARGET][-60:],
        mode='lines', name='История (60 дней)', line=dict(color='lightgray')
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['forecast'],
        mode='lines+markers', name='Прогноз (30 дней)', line=dict(color='red', width=3)
    ))
    fig.update_layout(
        title=f"Прогноз выручки на 30 дней: {business_name}",
        xaxis_title="Дата",
        yaxis_title="Выручка",
        hovermode="x unified"
    )
    plot_path = FORECAST_DIR / f"{business_name}_30d_forecast.html"
    fig.write_html(plot_path)
    
    print(f" Прогноз для {business_name} сохранён: {forecast_path.name}")
    print(f" График: {plot_path.name}")
    return forecast_df

def evaluate_models():
    """Оценка качества на последних 30 днях (hold-out)."""
    print("\n Оценка качества моделей на hold-out (последние 30 дней)...")
    report_lines = []
    report_lines.append("")

    for business in BUSINESSES.keys():
        try:
            df = pd.read_csv(FEATURES_DIR / f"{business}_features.csv", parse_dates=["date"])
            df = df.set_index("date").sort_index()
            if len(df) < 31:
                report_lines.append(f" {business}: недостаточно данных")
                continue

            holdout_df = df.iloc[-30:].copy()
            y_true = holdout_df[TARGET].values
            y_naive = df.iloc[-31:-1][TARGET].values

            model = joblib.load(MODELS_DIR / f"{business}_short_xgb.pkl")
            
            holdout_df['smoothed_trend'] = df[TARGET].rolling(window=14, min_periods=1).mean().ffill().bfill().iloc[-30:]
            holdout_df['month'] = holdout_df.index.month
            holdout_df['month_sin'] = np.sin(2 * np.pi * holdout_df['month'] / 12)
            holdout_df['month_cos'] = np.cos(2 * np.pi * holdout_df['month'] / 12)
            holdout_df['is_peak_month'] = holdout_df['month'].isin(BUSINESSES[business]["peak_months"]).astype(int)

            X_hold = holdout_df[SHORT_FEATURES_BASE].fillna(0)
            y_pred = model.predict(X_hold)
            y_pred = np.maximum(0, y_pred)

            metrics = calculate_metrics(y_true, y_pred, y_naive)
            report_lines.append(f" {business}")
            report_lines.append(f"    MAE:      {metrics['MAE']:,.0f}")
            report_lines.append(f"    RMSE:     {metrics['RMSE']:,.0f}")
            report_lines.append(f"    MAPE:     {metrics['MAPE (%)']:.2f}%")
            report_lines.append(f"    sMAPE:    {metrics['sMAPE (%)']:.2f}%")
            mase_status = " (лучше naive)" if metrics["MASE"] < 1 else " (хуже naive)"
            report_lines.append(f"    MASE:     {metrics['MASE']:.3f} {mase_status}")
            report_lines.append("")

        except Exception as e:
            report_lines.append(f" Ошибка для {business}: {e}")
            report_lines.append("")

    report_path = REPORT_DIR / "forecast_evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"\n Отчёт по метрикам сохранён: {report_path.relative_to(SRC_DIR)}")
    print("\n".join(report_lines))

def main():
    print(" Запуск обучения и прогнозирования на 30 дней")
    
    for name, config in BUSINESSES.items():
        file_path = FEATURES_DIR / f"{name}_features.csv"
        if not file_path.exists():
            print(f" Файл не найден: {file_path}")
            continue
        df = pd.read_csv(file_path, parse_dates=["date"])
        df = df.set_index("date").sort_index()
        train_and_save_models(df, name, config["peak_months"])
    
    for name in BUSINESSES.keys():
        try:
            forecast_30_days(name)
        except Exception as e:
            print(f" Ошибка при прогнозе для {name}: {e}")

    evaluate_models()
    
    print(f"\n Все результаты сохранены в: {FORECAST_DIR.relative_to(SRC_DIR)}")

if __name__ == "__main__":
    main()