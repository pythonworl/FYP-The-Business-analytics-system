import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

def forecast_sales(df, horizon=12, category=None):
    """
    Forecasting logic:
    1. Filter by category (optional)
    2. Aggregate Sales by Month
    3. Split Train/Test (last 6 months for test)
    4. Train multiple models (Holt-Winters, ARIMA, SMA)
    5. Evaluate on Test set (MAPE)
    6. Refit best model on ALL data and forecast 'horizon' months
    """
    
    # 1. Prepare Data
    df = df.copy()
    if "Order Date" not in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order_Date"], errors='coerce') # Handle potential naming differences
    else:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors='coerce')

    if category and category != "All":
        df = df[df["Category"] == category]
    
    # Aggregate monthly sales
    monthly_sales = df.set_index("Order Date").resample("M")["Sales"].sum()
    
    # Handle missing months (fill with 0 or interpolate)
    monthly_sales = monthly_sales.asfreq("M").fillna(0)

    if len(monthly_sales) < 12:
        return {"error": "Not enough data to forecast (need at least 12 months)"}

    # 2. Train/Test Split (Leave last 6 months for validation)
    test_size = 6
    train = monthly_sales[:-test_size]
    test = monthly_sales[-test_size:]

    results = {}

    # --- Model 1: Holt-Winters Exponential Smoothing ---
    try:
        hw_model = ExponentialSmoothing(train, seasonal="add", seasonal_periods=12, trend="add").fit()
        hw_pred = hw_model.forecast(test_size)
        results["Holt-Winters"] = mean_absolute_percentage_error(test, hw_pred) * 100
    except Exception as e:
        print(f"Holt-Winters failed: {e}")
        results["Holt-Winters"] = float("inf")

    # --- Model 2: ARIMA (Auto-Regressive Integrated Moving Average) ---
    try:
        # Simple order (1,1,1) for speed/stability. Auto-ARIMA is better but slower.
        arima_model = ARIMA(train, order=(1, 1, 1)).fit()
        arima_pred = arima_model.forecast(test_size)
        results["ARIMA"] = mean_absolute_percentage_error(test, arima_pred) * 100
    except Exception as e:
        print(f"ARIMA failed: {e}")
        results["ARIMA"] = float("inf")

    # --- Model 3: Simple Moving Average (3-month) ---
    try:
        # Naive forecast: last known SMA value repeated
        last_sma = train.rolling(window=3).mean().iloc[-1]
        sma_pred = pd.Series([last_sma] * test_size, index=test.index)
        results["SMA-3"] = mean_absolute_percentage_error(test, sma_pred) * 100
    except Exception as e:
        print(f"SMA failed: {e}")
        results["SMA-3"] = float("inf")

    # 3. Select Best Model
    best_model_name = min(results, key=results.get)
    best_mape = results[best_model_name]

    # 4. Refit Best Model on FULL Data & Forecast
    final_forecast = []
    
    # Historical data for chart
    history_dates = monthly_sales.index.strftime("%Y-%m").tolist()
    history_values = monthly_sales.values.tolist()

    # Future dates
    last_date = monthly_sales.index[-1]
    future_dates = [
        (last_date + pd.DateOffset(months=i+1)).strftime("%Y-%m") 
        for i in range(horizon)
    ]

    if best_model_name == "Holt-Winters":
        final_model = ExponentialSmoothing(monthly_sales, seasonal="add", seasonal_periods=12, trend="add").fit()
        final_values = final_model.forecast(horizon).tolist()
        
    elif best_model_name == "ARIMA":
        final_model = ARIMA(monthly_sales, order=(1, 1, 1)).fit()
        final_values = final_model.forecast(horizon).tolist()
        
    else: # SMA-3
        last_sma_val = monthly_sales.rolling(window=3).mean().iloc[-1]
        final_values = [last_sma_val] * horizon

    # Round values
    final_values = [round(x, 2) for x in final_values]
    history_values = [round(x, 2) for x in history_values]

    return {
        "model_name": best_model_name,
        "mape": round(best_mape, 2),
        "history": {
            "dates": history_dates,
            "values": history_values
        },
        "forecast": {
            "dates": future_dates,
            "values": final_values
        }
    }
