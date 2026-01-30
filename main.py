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

app = FastAPI(title="Business Analytics Predictor")

# Static + templates
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# Load models once
sales_model = joblib.load(SALES_MODEL_PATH)     # per-order sales
qty_model = joblib.load(QTY_MODEL_PATH)         # aggregated demand (monthly segment)

# Load dataset (for dropdowns + demand stats)
df = pd.read_csv(DATA_PATH)
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

# Keep only rows needed for demand/sales logic
needed_cols = ["Order Date", "Unit Price", "Discount", "Category", "Sub-Category", "Region", "City"]
df = df.dropna(subset=[c for c in needed_cols if c in df.columns]).copy()

df["Order_Year"] = df["Order Date"].dt.year.astype(int)
df["Order_Month"] = df["Order Date"].dt.month.astype(int)
df["Order_Quarter"] = df["Order Date"].dt.quarter.astype(int)

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
