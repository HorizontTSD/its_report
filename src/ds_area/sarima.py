
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

home_path = os.getcwd()
example_data_path = os.path.join(home_path, "src", "preprocess_data", "features", "Мойка_features.csv")

time_col = "date"
target_col = "Количество заказ-нарядов"

df_wash = pd.read_csv(example_data_path)
df_wash = df_wash[[time_col, target_col]]
df_wash[time_col] = pd.to_datetime(df_wash[time_col])
df_wash = df_wash.sort_values(time_col)

test_horizon = 31*3
df_train = df_wash.iloc[:-test_horizon].copy()
df_test = df_wash.iloc[-test_horizon:].copy()

df_train.set_index(time_col, inplace=True)
df_test.set_index(time_col, inplace=True)

df_train['month'] = df_train.index.month
df_train['dow'] = df_train.index.dayofweek
df_test['month'] = df_test.index.month
df_test['dow'] = df_test.index.dayofweek

auto_model = pm.auto_arima(
    df_train[target_col],
    exogenous=df_train[['month', 'dow']],
    seasonal=True,
    m=2,
    stepwise=True,
    suppress_warnings=True,
    trace=True
)
print(auto_model.summary())

sarima_model = SARIMAX(
    df_train[target_col],
    exog=df_train[['month', 'dow']],
    order=auto_model.order,
    seasonal_order=auto_model.seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_result = sarima_model.fit(disp=False)

forecast = sarima_result.get_forecast(
    steps=test_horizon,
    exog=df_test[['month', 'dow']]
)
forecast_df = forecast.summary_frame()
y_pred = forecast_df['mean'].to_list()
y_true = df_test[target_col].to_list()

print("MAE:", mean_absolute_error(y_true, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
print("R2:", r2_score(y_true, y_pred))
print("MAPE:", np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100)



import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(
    y=y_true,
    mode='lines',
    name='True',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    y=y_pred,
    mode='lines',
    name='Predicted',
    line=dict(color='orange')
))

fig.update_layout(
    title="Прогноз vs Реальные значения",
    xaxis_title="Время",
    yaxis_title=target_col,
    template="plotly_white",
    legend=dict(x=0.02, y=0.98)
)

fig.show()

