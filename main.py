from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pandas as pd
import joblib
from pathlib import Path


APP_DIR = Path(__file__).parent

DATA_PATH = APP_DIR / "Ecommerce_Sales_Data_Expanded.csv"
SALES_MODEL_PATH = APP_DIR / "best_sales_model.pkl"
QTY_MODEL_PATH = APP_DIR / "best_quantity_model.pkl"
CHURN_MODEL_PATH = APP_DIR / "best_churn_model.pkl"

# New Forecasting Module
from forecasting import forecast_sales
from chat_helper import ChatAssistant

app = FastAPI(title="Business Analytics Predictor")

# Static + templates
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# Load models once
sales_model = joblib.load(SALES_MODEL_PATH)     # per-order sales
qty_model = joblib.load(QTY_MODEL_PATH)         # aggregated demand (monthly segment)
churn_model = joblib.load(CHURN_MODEL_PATH)     # churn prediction

# Load dataset (for dropdowns + demand stats)
df = pd.read_csv(DATA_PATH)

# chat_bot initialized below after cleaning df

df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

# Keep only rows needed for demand/sales logic
needed_cols = ["Order Date", "Unit Price", "Discount", "Category", "Sub-Category", "Region", "City", "Sales", "Profit", "Quantity"]
df = df.dropna(subset=[c for c in needed_cols if c in df.columns]).copy()

df["Order_Year"] = df["Order Date"].dt.year.astype(int)
df["Order_Month"] = df["Order Date"].dt.month.astype(int)
df["Order_Quarter"] = df["Order Date"].dt.quarter.astype(int)

# Initialize Chat Assistant with cleaned data
chat_bot = ChatAssistant(df, sales_model, qty_model, churn_model)

# Dropdown lists (Category list is global; subcategories will be filtered via API)
CATEGORIES = sorted(df["Category"].dropna().unique().tolist())
SUBCATEGORIES = sorted(df["Sub-Category"].dropna().unique().tolist())
REGIONS = sorted(df["Region"].dropna().unique().tolist())
CITIES = sorted(df["City"].dropna().unique().tolist()) if "City" in df.columns else []

YEARS = sorted(df["Order_Year"].unique().tolist())
MONTHS = list(range(1, 13))

# ✅ Build monthly segment stats table (matches how the demand model was trained)
qty_agg = df.groupby(["Order_Year", "Order_Month", "Category", "Sub-Category", "Region"], as_index=False).agg(
    Avg_UnitPrice=("Unit Price", "mean"),
    Avg_Discount=("Discount", "mean"),
    Orders_Count=("Discount", "count")
)


def _segment_stats(category: str, sub_category: str, region: str, year: int, month: int):
    """
    Returns the aggregated numeric features used by the demand model:
      Avg_UnitPrice, Avg_Discount, Orders_Count
    Also returns stats_mode so you can verify if fallback is happening.
    """

    # 1) Exact match for that month/year/segment
    exact = qty_agg[
        (qty_agg["Category"] == category) &
        (qty_agg["Sub-Category"] == sub_category) &
        (qty_agg["Region"] == region) &
        (qty_agg["Order_Year"] == year) &
        (qty_agg["Order_Month"] == month)
    ]

    if len(exact) > 0:
        row = exact.iloc[0]
        return float(row["Avg_UnitPrice"]), float(row["Avg_Discount"]), int(row["Orders_Count"]), "exact_month"

    # 2) Segment fallback (same segment, any month/year)
    seg = qty_agg[
        (qty_agg["Category"] == category) &
        (qty_agg["Sub-Category"] == sub_category) &
        (qty_agg["Region"] == region)
    ]
    if len(seg) > 0:
        return float(seg["Avg_UnitPrice"].mean()), float(seg["Avg_Discount"].mean()), int(seg["Orders_Count"].mean()), "segment_fallback"

    # 3) Global fallback
    return float(df["Unit Price"].mean()), float(df["Discount"].mean()), int(len(df)), "global_fallback"


# ✅ NEW: API to get subcategories based on selected category
@app.get("/api/options/subcategories")
def get_subcategories(category: str = Query(...)):
    subset = df[df["Category"] == category]
    subs = sorted(subset["Sub-Category"].dropna().unique().tolist())
    return {"category": category, "subcategories": subs}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "categories": CATEGORIES,
            "subcategories": SUBCATEGORIES,   # initial list, will be replaced dynamically by app.js
            "regions": REGIONS,
            "cities": CITIES,
            "years": YEARS,
            "months": MONTHS,
        },
    )


@app.post("/api/predict/demand")
async def predict_demand(payload: dict):
    """
    Input:
      category, sub_category, region, year, month

    Output:
      predicted_total_quantity (integer),
      stats_mode,
      used_features
    """
    try:
        category = str(payload["category"])
        sub_category = str(payload["sub_category"])
        region = str(payload["region"])
        year = int(payload["year"])
        month = int(payload["month"])
    except Exception:
        return JSONResponse({"error": "Invalid payload for demand prediction."}, status_code=400)

    avg_price, avg_discount, orders_count, mode = _segment_stats(category, sub_category, region, year, month)

    # IMPORTANT: feature names must match training EXACTLY
    X = pd.DataFrame([{
        "Category": category,
        "Sub-Category": sub_category,
        "Region": region,
        "Order_Year": year,
        "Order_Month": month,
        "Avg_UnitPrice": avg_price,
        "Avg_Discount": avg_discount,
        "Orders_Count": orders_count
    }])

    pred = float(qty_model.predict(X)[0])

    # Demand is a count -> return integer + non-negative
    pred_int = int(round(pred))
    if pred_int < 0:
        pred_int = 0

    return {
        "predicted_total_quantity": pred_int,
        "stats_mode": mode,
        "used_features": {
            "Avg_UnitPrice": round(avg_price, 2),
            "Avg_Discount": round(avg_discount, 2),
            "Orders_Count": int(orders_count)
        }
    }


@app.post("/api/predict/sales")
async def predict_sales(payload: dict):
    """
    Per-order revenue prediction.
    Input (UI):
      category, sub_category, region, city,
      unit_price, discount, quantity

    NOTE: Year/Month/Quarter are auto-filled from the most recent date in the dataset
    to keep the Sales UI simple and avoid confusion.
    """
    try:
        latest_dt = df["Order Date"].dropna().max()
        auto_year = int(latest_dt.year)
        auto_month = int(latest_dt.month)
        auto_quarter = int(((auto_month - 1) // 3) + 1)

        X = pd.DataFrame([{
            "Category": str(payload["category"]),
            "Sub-Category": str(payload["sub_category"]),
            "Region": str(payload["region"]),
            "City": str(payload["city"]),
            "Unit Price": float(payload["unit_price"]),
            "Discount": float(payload["discount"]),
            "Order_Year": auto_year,
            "Order_Month": auto_month,
            "Order_Quarter": auto_quarter,
            "Quantity": float(payload["quantity"]),
        }])
    except Exception:
        return JSONResponse({"error": "Invalid payload for sales prediction."}, status_code=400)

    pred = float(sales_model.predict(X)[0])
    return {"predicted_sales": round(pred, 2)}


@app.post("/api/forecast/sales_series")
async def get_sales_forecast(payload: dict):
    """
    Input: { "horizon": 12, "category": "Furniture" (optional) }
    Output: JSON with history, forecast, and metrics.
    """
    try:
        horizon = int(payload.get("horizon", 12))
        category = payload.get("category", "All")
        
        result = forecast_sales(df, horizon=horizon, category=category)
        
        if "error" in result:
            return JSONResponse(result, status_code=400)
            
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ------------------------------------------------------------------------------
# REPORT GENERATION API
# ------------------------------------------------------------------------------

# Load Report Dataset
REPORT_DATA_PATH = APP_DIR / "women_clothing_ecommerce_sales.csv"
df_report = pd.read_csv(REPORT_DATA_PATH)
# Ensure date column is datetime
df_report["order_date"] = pd.to_datetime(df_report["order_date"], errors="coerce")
df_report["year"] = df_report["order_date"].dt.year
df_report["month"] = df_report["order_date"].dt.month

from insights_helper import generate_insights

@app.get("/api/report/summary")
def get_report_summary():
    """
    Returns aggregated business metrics for the report dashboard using 'women_clothing_ecommerce_sales.csv'.
    New Structure:
    1. KPIs: Revenue, Quantity, Orders, AOV (No Profit)
    2. Monthly Trend: Sales over time
    3. Size Share: Sales by Size
    4. Color Breakdown: Sales by Color
    5. Verbal Insights
    """
    try:
        # Use df_report for this endpoint
        
        # 1. KPIs
        total_sales = float(df_report["revenue"].sum())
        total_qty = int(df_report["quantity"].sum())
        total_orders = int(len(df_report))
        aov = total_sales / total_orders if total_orders > 0 else 0.0
        
        # 2. Monthly Trend (Group by Year-Month)
        trend_df = df_report.groupby(["year", "month"], as_index=False)["revenue"].sum()
        trend_df = trend_df.sort_values(by=["year", "month"])
        
        trend_labels = trend_df.apply(lambda x: f"{int(x['year'])}-{int(x['month']):02d}", axis=1).tolist()
        trend_sales = [float(x) for x in trend_df["revenue"].tolist()]

        # 3. Size Share
        size_df = df_report.groupby("size", as_index=False)["revenue"].sum().sort_values(by="revenue", ascending=False)
        size_labels = size_df["size"].astype(str).tolist()
        size_data = [float(x) for x in size_df["revenue"].tolist()]

        # 4. Color Distribution (Top 10)
        color_df = df_report.groupby("color", as_index=False)["revenue"].sum().sort_values(by="revenue", ascending=False).head(10)
        color_labels = color_df["color"].astype(str).tolist()
        color_data = [float(x) for x in color_df["revenue"].tolist()]

        data = {
            "kpis": {
                "total_sales": round(total_sales, 2),
                "total_quantity": total_qty,
                "total_orders": total_orders,
                "aov": round(float(aov), 2)
            },
            "trend": {
                "labels": trend_labels,
                "sales": [round(x, 2) for x in trend_sales]
            },
            "size_share": {
                "labels": size_labels,
                "data": [round(x, 2) for x in size_data]
            },
            "color_share": {
                "labels": color_labels,
                "data": [round(x, 2) for x in color_data]
            }
        }
        
        # 5. Generate Insights
        data["insights"] = generate_insights(data)
        
        return data

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/predict/churn")
async def predict_churn(payload: dict):
    """
    Predicts if a customer will make a purchase (PurchaseStatus=1) or not (PurchaseStatus=0).
    Input:
      Age, AnnualIncome, NumberOfPurchases, TimeSpentOnWebsite, 
      CustomerTenureYears, LastPurchaseDaysAgo, SessionCount,
      Gender, ProductCategory, PreferredDevice, Region, ReferralSource, CustomerSegment
    """
    try:
        # Create DataFrame from payload matching training features exactly
        X = pd.DataFrame([{
            "Age": float(payload["Age"]),
            "AnnualIncome": float(payload["AnnualIncome"]),
            "NumberOfPurchases": int(payload["NumberOfPurchases"]),
            "TimeSpentOnWebsite": float(payload["TimeSpentOnWebsite"]),
            "CustomerTenureYears": float(payload["CustomerTenureYears"]),
            "LastPurchaseDaysAgo": int(payload["LastPurchaseDaysAgo"]),
            "SessionCount": int(payload["SessionCount"]),
            "CustomerSatisfaction": int(payload["CustomerSatisfaction"]),
            "DiscountsAvailed": int(payload["DiscountsAvailed"]),
            "LoyaltyProgram": int(payload["LoyaltyProgram"])
        }])
        
        # Predict class and probability
        prediction = churn_model.predict(X)[0]
        probability = churn_model.predict_proba(X)[0][1] # Probability of Class 1 (Purchase)
        
        # Interpret result
        # Training target was "PurchaseStatus" (1=Purchase, 0=No Purchase/Churn)
        result_text = "Likely to Purchase" if prediction == 1 else "Likely to Churn"
        prob_percent = round(probability * 100, 2)
        
        return {
            "prediction": int(prediction),
            "result_text": result_text,
            "probability": prob_percent
        }
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/chat")
async def chat_with_bot(payload: dict):
    """
    Handles conversational queries for analytics, recommendations, and simulations.
    """
    try:
        message = payload.get("message", "")
        if not message:
            return JSONResponse({"error": "Empty message"}, status_code=400)
        
        response = chat_bot.handle_message(message)
        return response
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

