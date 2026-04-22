import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

def forecast_sales(df, horizon=12, category=None):
    """
    Forecasting logic:
    1. Filter by category (optional)
    2. Auto-Detect Frequency (Annual vs Monthly)
    3. Aggregate Sales by detected frequency
    4. Split Train/Test
    5. Train multiple models (Holt-Winters, ARIMA, SMA, Linear)
    6. Evaluate on Test set (MAE)
    7. Refit best model and forecast
    """
    
    # 1. Prepare Data
    df = df.copy()
    
    # Robust Column Mapping for Hydraulic Industry (revenue, units_sold)
    if "Sales" not in df.columns and "revenue" in df.columns:
        df["Sales"] = df["revenue"]
    if "Quantity" not in df.columns and "units_sold" in df.columns:
        df["Quantity"] = df["units_sold"]
        
    date_col = "Order Date" if "Order Date" in df.columns else ("Order_Date" if "Order_Date" in df.columns else None)
    if not date_col:
        return {"error": "Date column missing from data"}

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    if category and category != "All":
        if "Category" not in df.columns:
            return {"error": "Category column missing from data"}
        df = df[df["Category"] == category]
        if df.empty:
            return {"error": f"No data found for category: {category}"}
    
    if df.empty:
        return {"error": "No data points available for forecasting."}

    # 1.5 Auto-Detect Frequency
    unique_months = df[date_col].dt.month.nunique()
    unique_years = df[date_col].dt.year.nunique()
    
    # If data is sparse (only one month ever mentioned, e.g. January 1st for multiple years)
    if unique_months == 1 and unique_years >= 2:
        freq = 'AS' # Annual Start
        horizon_points = max(1, int(round(horizon / 12)))
        human_freq = "Yearly"
        min_points = 3
        test_size = 1
    else:
        freq = 'M' # Monthly
        horizon_points = int(horizon)
        human_freq = "Monthly"
        min_points = 12
        test_size = 6

    # Aggregate sales (STRICT GROUPING)
    series = df.set_index(date_col).resample(freq)["Sales"].sum()
    series = series.asfreq(freq).fillna(0)

    # 1.6 PARTIAL MONTH FILTER (The 'Down and Down' Fix)
    # If the last month has < 20% of the average volume, it's likely a partial month.
    # We remove it to prevent it from dragging the forecast down.
    if len(series) > 3:
        avg_vol = series.iloc[:-1].mean()
        if series.iloc[-1] < (avg_vol * 0.2):
            series = series.iloc[:-1]

    if len(series) < min_points:
        return {"error": f"Insufficient {human_freq} data. Found {len(series)} points, need at least {min_points}."}

    # 2. Train/Test Split
    # Extremely lenient split: need at least 4 points to validate, otherwise just train on all.
    if len(series) >= 4:
        test_size = max(1, len(series) // 4)
        train = series[:-test_size]
        test = series[-test_size:]
    else:
        train = series
        test = series # Dummy test on same data
        test_size = 0

    results = {}

    # --- Model 1: Holt-Winters (ENHANCED SENSITIVITY) ---
    try:
        # Try Multiplicative if possible (better for growth models), fallback to Additive
        if freq == 'M' and len(train) >= 24:
            try:
                hw_model = ExponentialSmoothing(train, seasonal="mul", seasonal_periods=12, trend="add", damped_trend=True).fit()
            except:
                hw_model = ExponentialSmoothing(train, seasonal="add", seasonal_periods=12, trend="add").fit()
        else:
            hw_model = ExponentialSmoothing(train, trend="add", damped_trend=True).fit()
            
        hw_pred = hw_model.forecast(test_size)
        results["Holt-Winters"] = mean_absolute_error(test, hw_pred)
    except Exception:
        results["Holt-Winters"] = float("inf")

    # --- Model 2: ARIMA (ROBUST ORDER) ---
    try:
        # ARIMA(2,1,2) captures 'wave' patterns better than (1,1,1)
        arima_model = ARIMA(train, order=(2, 1, 1)).fit()
        arima_pred = arima_model.forecast(test_size)
        results["ARIMA"] = mean_absolute_error(test, arima_pred)
    except Exception:
        results["ARIMA"] = float("inf")

    # --- Model 3: Linear Trend (PENALIZED) ---
    try:
        from sklearn.linear_model import LinearRegression
        X_train = np.arange(len(train)).reshape(-1, 1)
        lr = LinearRegression().fit(X_train, train.values)
        X_test = np.arange(len(train), len(train) + test_size).reshape(-1, 1)
        lr_pred = lr.predict(X_test)
        # Linear trend is often too 'flat' or 'dumb', so we penalize it to favor seasonality
        results["Linear-Trend"] = mean_absolute_error(test, lr_pred) * 1.8
    except Exception:
        results["Linear-Trend"] = float("inf")

    # --- Model 4: SMA (DE-PRIORITIZED) ---
    try:
        win = min(6, len(train)) # Longer window for stability, but high penalty
        last_sma = train.rolling(window=win).mean().iloc[-1]
        sma_pred = pd.Series([last_sma] * test_size, index=test.index)
        results["SMA"] = mean_absolute_error(test, sma_pred) * 2.5
    except Exception:
        results["SMA"] = float("inf")

    # 3. Select Best Model
    valid_results = {k: v for k, v in results.items() if v != float("inf") and not np.isnan(v)}
    if not valid_results:
        best_model_name = "Naive (Last Value)"
        best_mae = 0.0
    else:
        best_model_name = min(valid_results, key=valid_results.get)
        best_mae = valid_results[best_model_name]

    # 4. Refit and Forecast (QUALITY FIRST)
    ui_series = series.tail(36)
    history_dates = ui_series.index.strftime("%Y" if freq == 'AS' else "%Y-%m").tolist()
    history_values = ui_series.values.tolist()
    last_date = series.index[-1]
    
    if freq == 'AS':
        future_dates = [(last_date + pd.DateOffset(years=i+1)).strftime("%Y") for i in range(horizon_points)]
    else:
        future_dates = [(last_date + pd.DateOffset(months=i+1)).strftime("%Y-%m") for i in range(horizon_points)]

    if best_model_name == "Holt-Winters":
        if freq == 'M' and len(series) >= 24:
            try:
                final_model = ExponentialSmoothing(series, seasonal="mul", seasonal_periods=12, trend="add", damped_trend=True).fit()
            except:
                final_model = ExponentialSmoothing(series, seasonal="add", seasonal_periods=12, trend="add").fit()
        else:
            final_model = ExponentialSmoothing(series, trend="add", damped_trend=True).fit()
        final_values = final_model.forecast(horizon_points).tolist()
    elif best_model_name == "ARIMA":
        final_model = ARIMA(series, order=(2, 1, 1)).fit()
        final_values = final_model.forecast(horizon_points).tolist()
    elif best_model_name == "Linear-Trend":
        X_all = np.arange(len(series)).reshape(-1, 1)
        lr_final = LinearRegression().fit(X_all, series.values)
        X_future = np.arange(len(series), len(series) + horizon_points).reshape(-1, 1)
        final_values = lr_final.predict(X_future).tolist()
    else: # SMA / Naive
        win = min(6, len(series))
        last_sma_val = series.rolling(window=win).mean().iloc[-1]
        final_values = [last_sma_val] * horizon_points

    # Final Polish: Sensitivity Blend
    # If the forecast is too flat (std < 1% of mean), we inject a safe amount of momentum
    if np.std(final_values) < (np.mean(final_values) * 0.01):
        # Add 0.5% monthly growth or decay based on the last year's trend
        last_year_growth = (series.iloc[-1] / series.iloc[-12] - 1) / 12 if len(series) >= 12 else 0
        final_values = [v * (1 + last_year_growth * (i+1)) for i, v in enumerate(final_values)]

    # Ensure no negative predictions and round
    final_values = [round(float(max(0, x)), 2) for x in final_values]
    history_values = [round(float(max(0, x)), 2) for x in history_values]

    return {
        "model_name": best_model_name,
        "mae": round(best_mae, 2),
        "freq": human_freq,
        "history": { "dates": history_dates, "values": history_values },
        "forecast": { "dates": future_dates, "values": final_values }
    }
