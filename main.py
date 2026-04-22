from fastapi import FastAPI, Request, Query, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import shutil
import logging
import os
from dotenv import load_dotenv

# Load environment variables (for Gemini API Key)
load_dotenv()

from trainer import (
    train_user_sales_and_quantity_models,
    train_user_churn_model,
    get_user_model_dir,
    get_user_data_dir,
    generate_sample_csv,
    generate_batch_predictions,
    fuzzy_column_mapper,
    get_feature_metadata
)

APP_DIR = Path(__file__).parent

# Default Global Data and Models
DATA_PATH = APP_DIR / "Ecommerce_Sales_Data_Expanded.csv"
SALES_MODEL_PATH = APP_DIR / "best_sales_model.pkl"
QTY_MODEL_PATH = APP_DIR / "best_quantity_model.pkl"
CHURN_MODEL_PATH = APP_DIR / "best_churn_model.pkl"
REPORT_DATA_PATH = APP_DIR / "women_clothing_ecommerce_sales.csv"

# Existing modules
from forecasting import forecast_sales
from chat_helper import ChatAssistant
from insights_helper import generate_insights

app = FastAPI(title="Business Analytics Predictor")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "fyp_secret_key_12345"))

app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# Preload global defaults
default_sales_model = joblib.load(SALES_MODEL_PATH)
default_qty_model = joblib.load(QTY_MODEL_PATH)
default_churn_model = joblib.load(CHURN_MODEL_PATH)
default_df = pd.read_csv(DATA_PATH)

# AutoML Status Tracker (In-Memory)
automl_jobs = {}

# Performance Caching
user_data_cache = {}    # Stores DataFrames to prevent repeated CSV reading
user_options_cache = {} # Stores dropdown lists (categories, regions)
# Helpers for User-Specific resolution
# ==========================================
def check_user_has_data(username: str) -> bool:
    if not username:
        return False
    data_dir = get_user_data_dir(username)
    return (data_dir / "sales_data.csv").exists() or (data_dir / "churn_data.csv").exists() or (data_dir / "report_data.csv").exists()

def invalidate_user_automl(username: str):
    """
    Deletes the current AutoML models and metadata to ensure a clean slate.
    Called when new data is uploaded or generated.
    """
    model_dir = get_user_model_dir(username)
    data_dir = get_user_data_dir(username)
    
    # Strictly wipe the model directory to prevent stale model 'leakage'
    if model_dir.exists():
        for f in model_dir.glob("*"):
            if f.is_file():
                f.unlink()
    
    # Remove old prediction reports
    for f in data_dir.glob("*_batch_predictions.csv"):
        f.unlink()
    
    if username in automl_jobs:
        del automl_jobs[username]
    
    # Clear caches
    if username in user_data_cache: del user_data_cache[username]
    if username in user_options_cache: del user_options_cache[username]

def get_user_dataframe(username: str) -> pd.DataFrame:
    # Use Cache if available to prevent 32MB file reading lag
    if username in user_data_cache:
        return user_data_cache[username]

    if username:
        data_dir = get_user_data_dir(username)
        sales_file = data_dir / "sales_data.csv"
        if sales_file.exists():
            logging.info(f"Loading large dataset for {username}...")
            user_df = pd.read_csv(sales_file)
            # 1. Cleaning & Stripping Spaces globally
            user_df.columns = [str(c).strip() for c in user_df.columns]
            # Added mappings for Hydraulic Industry (product_type, units_sold, revenue)
            user_df = fuzzy_column_mapper(user_df, ["Order Date", "Date", "Unit Price", "Discount", "Category", "Sub-Category", "Region", "City", "Sales", "Profit", "Quantity", "product_type", "units_sold", "revenue"])
        else:
            user_df = default_df.copy()
    else:
        user_df = default_df.copy()

    # Column Fallbacks for UI
    if "Category" not in user_df.columns: user_df["Category"] = "General"
    if "Sub-Category" not in user_df.columns: user_df["Sub-Category"] = "General"

    if "Order Date" in user_df.columns:
        # Robust year detection (detects 2020 as 2020, not 1970)
        def robust_year_detect(val):
            try:
                # If it's a number like 2022
                if isinstance(val, (int, float)) and 1900 < val < 2100: return int(val)
                # If it's a string that's just a number "2022"
                s = str(val).strip()
                if s.isdigit() and 1900 < int(s) < 2100: return int(s)
                # Fallback to datetime parsing
                dt = pd.to_datetime(val, errors="coerce")
                return dt.year if pd.notnull(dt) else 2025
            except: return 2025
        
        user_df["Order_Year"] = user_df["Order Date"].apply(robust_year_detect)
        user_df["Order Date"] = pd.to_datetime(user_df["Order Date"], errors="coerce")
        user_df["Order_Month"] = user_df["Order Date"].dt.month.fillna(1).astype(int)
        user_df["Order_Quarter"] = user_df["Order Date"].dt.quarter.fillna(1).astype(int)

    # Standardize Category names for robust matching (e.g. Huracn -> Huracan)
    if "Category" in user_df.columns:
        user_df["Category"] = user_df["Category"].astype(str).str.replace(r'[^\x00-\x7F]+', 'a', regex=True).str.strip()
        user_df["Category"] = user_df["Category"].apply(lambda x: "".join([c for c in x if ord(c) < 128]))

    # Store in cache
    if username:
        user_data_cache[username] = user_df

    return user_df

def get_report_dataframe(username: str, source: str = "sales") -> pd.DataFrame:
    if username:
        data_dir = get_user_data_dir(username)
        if source == "sales":
            report_file = data_dir / "sales_data.csv"
        elif source == "churn":
            report_file = data_dir / "churn_data.csv"
        else:
            report_file = data_dir / "report_data.csv"
            
        if report_file.exists():
            df = pd.read_csv(report_file)
            # Global Column Cleaning
            df.columns = [str(c).strip() for c in df.columns]
            
            # Apply Fuzzy Mapping for reporting consistency
            if source == "sales":
                df = fuzzy_column_mapper(df, ["Order Date", "Sales", "Quantity", "Unit Price", "Discount", "Category", "Sub-Category", "Region"])
            elif source == "churn":
                df = fuzzy_column_mapper(df, ["PurchaseStatus", "Age", "AnnualIncome", "NumberOfPurchases", "CustomerSatisfaction"])
            
            if source == "report" and "order_date" in df.columns:
                df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
                df["year"] = df["order_date"].dt.year
                df["month"] = df["order_date"].dt.month
            return df
            
    # Fallback to default datasets ONLY for anonymous visitors
    if not username:
        if source == "sales":
            return pd.read_csv(DATA_PATH)
        elif source == "churn":
            return pd.read_csv(APP_DIR / "customerData_500k.csv")
        else:
            df = pd.read_csv(REPORT_DATA_PATH)
            if "order_date" in df.columns:
                df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
                df["year"] = df["order_date"].dt.year
                df["month"] = df["order_date"].dt.month
            return df
    
    return None

def get_user_models(username: str):
    """
    Returns (sales_model, qty_model, churn_model).
    If a personalized model exists, it is loaded.
    If not, it returns None to enforce strict data-model isolation.
    """
    if not username:
        return default_sales_model, default_qty_model, default_churn_model
    
    model_dir = get_user_model_dir(username)
    sm_path = model_dir / "best_sales_model.pkl"
    qm_path = model_dir / "best_quantity_model.pkl"
    cm_path = model_dir / "best_churn_model.pkl"
    
    sm = joblib.load(sm_path) if sm_path.exists() else None
    qm = joblib.load(qm_path) if qm_path.exists() else None
    cm = joblib.load(cm_path) if cm_path.exists() else None
    return sm, qm, cm

def get_dropdowns(user_df: pd.DataFrame, username: str = None):
    # Use cache for options to prevent UI lag on refresh
    if username and username in user_options_cache:
        return user_options_cache[username]
        
    cat = sorted(user_df["Category"].dropna().unique().tolist()) if "Category" in user_df.columns else []
    sub = sorted(user_df["Sub-Category"].dropna().unique().tolist()) if "Sub-Category" in user_df.columns else []
    reg = sorted(user_df["Region"].dropna().unique().tolist()) if "Region" in user_df.columns else []
    cit = sorted(user_df["City"].dropna().unique().tolist()) if "City" in user_df.columns else []
    yrs = sorted(user_df["Order_Year"].unique().tolist()) if "Order_Year" in user_df.columns else [2021, 2022, 2023, 2024, 2025]
    mth = list(range(1, 13))
    
    res = (cat, sub, reg, cit, yrs, mth)
    if username:
        user_options_cache[username] = res
    return res

def get_segment_stats(df_local, category, sub_category, region, year, month):
    if "Order_Year" not in df_local.columns:
        return 0, 0, 0, "no_data"
        
    qty_agg = df_local.groupby(["Order_Year", "Order_Month", "Category", "Sub-Category", "Region"], as_index=False).agg(
        Avg_UnitPrice=("Unit Price", "mean"),
        Avg_Discount=("Discount", "mean"),
        Total_Quantity=("Quantity", "sum")
    )
    # Use category-wide total volume (Initial Logic)
    avg_annual_qty = df_local[df_local["Category"].str.lower() == str(category).lower()]["Quantity"].sum()
    
    # 3. DATA-DRIVEN SEASONALITY (Hydraulic Calibration)
    # We remove artificial multipliers and use the real data averages.
    
    # We can detect month if available
    m = int(month)
    
    # Simple statistical average per month for this category
    # If the user provides monthly rows, the AI model will pick up the real trend.
    # For now, we return the baseline average from the dataset.
    final_avg = (avg_annual_qty / 12) if avg_annual_qty > 0 else 1
    
    return float(df_local["Unit Price"].mean()), float(df_local["Discount"].mean()), max(1, int(round(final_avg))), "statistical_average"


# ==========================================
# AUTH ROUTES
# ==========================================
@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login_post(request: Request, username: str = Form(...)):
    if username.strip():
        request.session["username"] = username.strip()
    return RedirectResponse(url="/", status_code=303)

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    response = RedirectResponse(url="/login")
    response.delete_cookie("session") # Hard wipe session cookie
    return response

@app.post("/api/reset_workspace")
async def reset_workspace_route(request: Request):
    username = request.session.get("username")
    if not username: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    invalidate_user_automl(username)
    # Also wipe data files for a truly "Fresh" start if requested
    data_dir = get_user_data_dir(username)
    for f in data_dir.glob("*.csv"):
        f.unlink()
    return {"message": "Workspace successfully reset. You are back to a clean slate!"}

# ==========================================
# UPLOAD ROUTES (JUST SAVING NOW)
# ==========================================
@app.post("/api/upload/sales")
async def api_upload_sales(request: Request, file: UploadFile = File(...)):
    username = request.session.get("username")
    if not username: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    data_dir = get_user_data_dir(username)
    file_path = data_dir / "sales_data.csv"
    
    # Isolation: Delete Churn data when Sales is uploaded
    churn_file = data_dir / "churn_data.csv"
    if churn_file.exists(): churn_file.unlink()
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    invalidate_user_automl(username)
    return {"message": "Sales data uploaded! (Previous Churn data cleared)"}

@app.post("/api/upload/churn")
async def api_upload_churn(request: Request, file: UploadFile = File(...)):
    username = request.session.get("username")
    if not username: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    data_dir = get_user_data_dir(username)
    file_path = data_dir / "churn_data.csv"
    
    # Isolation: Delete Sales data when Churn is uploaded
    sales_file = data_dir / "sales_data.csv"
    if sales_file.exists(): sales_file.unlink()
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    invalidate_user_automl(username)
    return {"message": "Churn data uploaded! (Previous Sales data cleared)"}

@app.post("/api/upload/report")
async def api_upload_report(request: Request, file: UploadFile = File(...)):
    username = request.session.get("username")
    if not username: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    data_dir = get_user_data_dir(username)
    file_path = data_dir / "report_data.csv"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "Custom Business Report Data uploaded successfully!"}

# ==========================================
# AUTO ML & SAMPLE GENERATION
# ==========================================
@app.post("/api/generate-sample")
async def api_gen_sample(request: Request, payload: dict):
    username = request.session.get("username")
    if not username: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    cat = payload.get("category", "Ice Cream")
    dtype = payload.get("type", "sales")
    
    data_dir = get_user_data_dir(username)
    if dtype == "sales":
        rival = data_dir / "churn_data.csv"
    else:
        rival = data_dir / "sales_data.csv"
        
    if rival.exists(): rival.unlink()
    
    path = generate_sample_csv(username, cat, data_type=dtype)
    invalidate_user_automl(username)
    return {"message": f"Generated {dtype} data for {cat}! (Other data cleared)", "file": path}

def background_automl_task(username: str):
    import json
    try:
        automl_jobs[username] = {"status": "processing", "progress": 10}
        data_dir = get_user_data_dir(username)
        model_dir = get_user_model_dir(username)
        
        all_meta = {}

        # 1. Sales & Demand
        sales_file = data_dir / "sales_data.csv"
        if sales_file.exists():
            df_s = pd.read_csv(sales_file)
            res_s = train_user_sales_and_quantity_models(username, df_s)
            all_meta["sales_demand"] = res_s
        automl_jobs[username]["progress"] = 50
        
        # 2. Churn
        churn_file = data_dir / "churn_data.csv"
        if churn_file.exists():
            df_c = pd.read_csv(churn_file)
            res_c = train_user_churn_model(username, df_c)
            all_meta["churn"] = res_c
        automl_jobs[username]["progress"] = 80
        
        # 3. Save Metadata
        with open(model_dir / "model_metadata.json", "w") as f:
            json.dump(all_meta, f)

        # 4. Batch Predictions
        generate_batch_predictions(username)
        
        automl_jobs[username] = {"status": "completed", "progress": 100}
    except Exception as e:
        logging.error(f"AutoML Error for {username}: {e}")
        automl_jobs[username] = {"status": "failed", "error": str(e)}

@app.post("/api/run_automl")
async def run_automl_route(request: Request, background_tasks: BackgroundTasks):
    username = request.session.get("username")
    if not username: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    if not check_user_has_data(username):
        return JSONResponse({"error": "No data found. Please upload or generate sample data first."}, status_code=400)
    
    background_tasks.add_task(background_automl_task, username)
    return {"message": "AutoML Pipeline started in background."}

@app.get("/api/automl_status")
async def get_automl_status(request: Request):
    username = request.session.get("username")
    status = automl_jobs.get(username, {"status": "idle"})
    return status

@app.get("/api/automl_metrics")
async def get_automl_metrics(request: Request):
    import json
    username = request.session.get("username")
    if not username: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    model_dir = get_user_model_dir(username)
    data_dir = get_user_data_dir(username)
    pred_path_s = data_dir / "sales_batch_predictions.csv"
    pred_path_c = data_dir / "churn_batch_predictions.csv"
    
    metrics = {}
    meta_path = model_dir / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metrics = json.load(f)
            
    return {
        "metrics": metrics,
        "has_sales_preds": pred_path_s.exists(),
        "has_churn_preds": pred_path_c.exists(),
        "has_predictions": pred_path_s.exists() or pred_path_c.exists()
    }
@app.get("/api/user_model_status")
async def get_user_model_status(request: Request):
    username = request.session.get("username")
    if not username: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    model_dir = get_user_model_dir(username)
    data_dir = get_user_data_dir(username)
    
    return {
        "has_sales_data": (data_dir / "sales_data.csv").exists(),
        "has_churn_data": (data_dir / "churn_data.csv").exists(),
        "has_sales_model": (model_dir / "best_sales_model.pkl").exists(),
        "has_qty_model": (model_dir / "best_quantity_model.pkl").exists(),
        "has_churn_model": (model_dir / "best_churn_model.pkl").exists()
    }

@app.get("/api/download/predictions")
async def download_predictions_route(request: Request, type: str = Query("sales")):
    username = request.session.get("username")
    if not username: return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    data_dir = get_user_data_dir(username)
    if type == "churn":
        file_path = data_dir / "churn_batch_predictions.csv"
        filename = "churn_predictions_report.csv"
    else:
        file_path = data_dir / "sales_batch_predictions.csv"
        filename = "sales_predictions_report.csv"

    if not file_path.exists():
        return JSONResponse({"error": f"No {type} prediction file found. Run AutoML first."}, status_code=404)
        
    return FileResponse(path=file_path, filename=filename)

# ==========================================
# MAIN UI ROUTE
# ==========================================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    username = request.session.get("username")
    if not username:
        return RedirectResponse(url="/login", status_code=303)

    user_df = get_user_dataframe(username)
    categories, subcategories, regions, cities, years, months = get_dropdowns(user_df)

    latest_year = years[-1] if years else 2025
    is_demo_mode = not check_user_has_data(username)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "username": username,
            "is_demo_mode": is_demo_mode,
            "categories": categories,
            "subcategories": subcategories,
            "regions": regions,
            "cities": cities,
            "years": years,
            "months": months,
            "latest_year": latest_year
        },
    )

@app.get("/api/options/all")
def get_all_options(request: Request):
    username = request.session.get("username")
    user_df = get_user_dataframe(username)
    cat, sub, reg, cit, yrs, mth = get_dropdowns(user_df)
    return {
        "categories": cat,
        "subcategories": sub,
        "regions": reg,
        "cities": cit,
        "years": yrs,
        "months": mth
    }

@app.get("/api/options/subcategories")
def get_subcategories(request: Request, category: str = Query(...)):
    username = request.session.get("username")
    user_df = get_user_dataframe(username)
    if "Category" not in user_df.columns: return {"category": category, "subcategories": []}
    subset = user_df[user_df["Category"] == category]
    subs = sorted(subset["Sub-Category"].dropna().unique().tolist()) if "Sub-Category" in subset.columns else []
    return {"category": category, "subcategories": subs}

# ==========================================
# PREDICTION ROUTES
# ==========================================
@app.post("/api/predict/demand")
async def predict_demand(request: Request, payload: dict):
    username = request.session.get("username")
    user_df = get_user_dataframe(username)
    _, qty_model, _ = get_user_models(username)
    
    try:
        # Normalize input to match standardized database names
        category = "".join([c for c in str(payload.get("category", "All")) if ord(c) < 128])
        sub_category = "".join([c for c in str(payload.get("sub_category", "General")) if ord(c) < 128])
        region = payload.get("region", "All")
        year = payload.get("year", 2025)
        month = payload.get("month", 1)
    except Exception:
        return JSONResponse({"error": "Invalid payload for demand prediction."}, status_code=400)

    avg_price, avg_discount, orders_count, mode = get_segment_stats(user_df, category, sub_category, region, year, month)

    X = pd.DataFrame([{
        "Category": category,
        "Sub-Category": sub_category,
        "Region": region,
        "Order_Year": int(year),
        "Order_Month": int(month),
        "Avg_UnitPrice": float(avg_price),
        "Avg_Discount": float(avg_discount)
    }])

    # 1. Use the ML model if available (RESTORED DIRECT SENSITIVITY)
    if qty_model:
        try:
            pred = float(qty_model.predict(X)[0])
            # Removed blending to ensure the number responds directly to Month/Season
            return {
                "predicted_total_quantity": int(round(pred * 1.5 if orders_count > 0 else pred)),
                "model_source": "Model: Personalized Business Data",
                "stats_mode": mode
            }
        except Exception as e:
            logging.error(f"Prediction Error: {e}")
            return {"predicted_total_quantity": int(orders_count), "stats_mode": mode, "model_source": "Baseline Stats"}
    
    # 2. If no model, use the seasonal average calculated in get_segment_stats
    return {"predicted_total_quantity": int(orders_count), "stats_mode": mode, "model_source": "Baseline Stats"}

@app.post("/api/predict/sales")
async def predict_sales(request: Request, payload: dict):
    username = request.session.get("username")
    user_df = get_user_dataframe(username)
    sales_model, _, _ = get_user_models(username)
    
    if not sales_model:
        return JSONResponse({"error": "No personalized sales model found. Please upload/generate sales data and run AutoML first."}, status_code=400)
    
    try:
        import json as json_mod
        model_dir = get_user_model_dir(username)
        config_path = model_dir / "sales_feature_config.json"
        
        # Auto-detect date features from user data
        if "Order Date" in user_df.columns:
            latest_dt = user_df["Order Date"].dropna().max()
            auto_year = int(latest_dt.year) if pd.notnull(latest_dt) else 2025
            auto_month = int(latest_dt.month) if pd.notnull(latest_dt) else 1
            auto_quarter = int(((auto_month - 1) // 3) + 1) if pd.notnull(latest_dt) else 1
        else:
            auto_year, auto_month, auto_quarter = 2025, 1, 1
        
        if config_path.exists():
            # === DYNAMIC MODE ===
            with open(config_path, "r") as f:
                feature_config = json_mod.load(f)
            
            numerical_features = feature_config.get("numerical_features", [])
            categorical_features = feature_config.get("categorical_features", [])
            all_features = feature_config.get("all_features", numerical_features + categorical_features)
            
            # Auto-fill date-derived features
            auto_fill = {
                "Order_Year": auto_year,
                "Order_Month": auto_month,
                "Order_Quarter": auto_quarter
            }
            
            row = {}
            for feat in all_features:
                # Check auto-fill first
                if feat in auto_fill:
                    row[feat] = auto_fill[feat]
                    continue
                
                val = payload.get(feat)
                if feat in numerical_features:
                    if val is None or str(val).strip() == "":
                        row[feat] = np.nan
                    else:
                        try:
                            row[feat] = float(val)
                        except:
                            row[feat] = np.nan
                elif feat in categorical_features:
                    row[feat] = str(val) if val is not None and str(val).strip() != "" else "Unknown"
            
            X = pd.DataFrame([row])
            for feat in all_features:
                if feat not in X.columns:
                    X[feat] = np.nan
            X = X[all_features]
        else:
            # === LEGACY FALLBACK ===
            X = pd.DataFrame([{
                "Category": str(payload.get("category", "General")),
                "Sub-Category": str(payload.get("sub_category", "General")),
                "Region": str(payload.get("region", "Global")),
                "Unit Price": float(payload.get("unit_price", 100)),
                "Discount": float(payload.get("discount", 0)),
                "Order_Year": auto_year,
                "Order_Month": auto_month,
                "Order_Quarter": auto_quarter,
                "Quantity": float(payload.get("quantity", 1)),
            }])
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    try:
        pred = float(sales_model.predict(X)[0])
    except Exception as e:
        return JSONResponse({"error": f"Prediction failed: {e}"}, status_code=500)
    
    return {
        "predicted_sales": round(pred, 2),
        "model_source": "Personalized Business Data" if (get_user_model_dir(username) / "best_sales_model.pkl").exists() else "General Demo Data"
    }

@app.post("/api/forecast/sales_series")
async def api_get_sales_forecast(request: Request, payload: dict):
    username = request.session.get("username")
    user_df = get_user_dataframe(username)
    
    try:
        horizon = int(payload.get("horizon", 12))
        category = payload.get("category", "All")
        result = forecast_sales(user_df, horizon=horizon, category=category)
        if "error" in result:
            return JSONResponse(result, status_code=400)
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/sales/features")
async def get_sales_features(request: Request):
    """Returns the dynamically detected features for the sales model."""
    username = request.session.get("username")
    if not username:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    metadata = get_feature_metadata(username, model_type="sales")
    if metadata:
        return {"has_features": True, **metadata}
    else:
        return {"has_features": False, "message": "No sales model trained yet. Run AutoML first."}

@app.get("/api/churn/features")
async def get_churn_features(request: Request):
    """Returns the dynamically detected features for the churn model."""
    username = request.session.get("username")
    if not username:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    metadata = get_feature_metadata(username, model_type="churn")
    if metadata:
        return {"has_features": True, **metadata}
    else:
        return {"has_features": False, "message": "No churn model trained yet. Run AutoML first."}

@app.post("/api/predict/churn")
async def predict_churn(request: Request, payload: dict):
    username = request.session.get("username")
    _, _, churn_model = get_user_models(username)
    
    if not churn_model:
        return JSONResponse({"error": "No personalized churn model found. Please upload/generate churn data and run AutoML first."}, status_code=400)
    
    try:
        import json as json_mod
        model_dir = get_user_model_dir(username)
        config_path = model_dir / "churn_feature_config.json"
        
        if config_path.exists():
            # === DYNAMIC MODE: Use saved feature config ===
            with open(config_path, "r") as f:
                feature_config = json_mod.load(f)
            
            numerical_features = feature_config.get("numerical_features", [])
            categorical_features = feature_config.get("categorical_features", [])
            all_features = feature_config.get("all_features", numerical_features + categorical_features)
            
            row = {}
            for feat in all_features:
                val = payload.get(feat)
                if feat in numerical_features:
                    if val is None or str(val).strip() == "":
                        row[feat] = np.nan
                    else:
                        try:
                            row[feat] = float(val)
                        except:
                            row[feat] = np.nan
                elif feat in categorical_features:
                    row[feat] = str(val) if val is not None and str(val).strip() != "" else np.nan
            
            X = pd.DataFrame([row])
            # Ensure column order matches training
            for feat in all_features:
                if feat not in X.columns:
                    X[feat] = np.nan
            X = X[all_features]
        else:
            # === LEGACY FALLBACK: Hardcoded features for old models ===
            def to_float(val):
                if val is None or str(val).strip() == "": return np.nan
                try: return float(val)
                except: return np.nan

            X = pd.DataFrame([{
                "NumberOfPurchases": int(payload.get("NumberOfPurchases", 5)),
                "CustomerTenureYears": to_float(payload.get("CustomerTenureYears")),
                "LastPurchaseDaysAgo": int(payload.get("LastPurchaseDaysAgo", 30)),
                "CustomerSatisfaction": int(payload.get("CustomerSatisfaction", 3)),
                "DiscountsAvailed": to_float(payload.get("DiscountsAvailed")),
                "LoyaltyProgram": to_float(payload.get("LoyaltyProgram"))
            }])

        prediction = int(churn_model.predict(X)[0])
        
        # Determine label semantics from saved config
        is_churn_inverted = False
        if config_path.exists():
            is_churn_inverted = feature_config.get("is_churn_inverted", False)
        
        # Get probabilities
        if hasattr(churn_model, "predict_proba"):
            proba_arr = churn_model.predict_proba(X)[0]
            
            if is_churn_inverted:
                # Churn-style: class 1 = churned, class 0 = stayed
                churn_probability = float(proba_arr[1]) if len(proba_arr) > 1 else float(proba_arr[0])
                result_text = "Likely to Churn" if prediction == 1 else "Likely to Stay"
                # Always show churn risk probability
                prob_percent = round(churn_probability * 100, 2)
            else:
                # PurchaseStatus-style: class 1 = purchased, class 0 = churned
                churn_probability = float(proba_arr[0]) if len(proba_arr) > 1 else 1.0 - float(proba_arr[0])
                result_text = "Likely to Purchase" if prediction == 1 else "Likely to Churn"
                prob_percent = round(float(proba_arr[prediction]) * 100, 2)
        else:
            prob_percent = 50.0
            if is_churn_inverted:
                result_text = "Likely to Churn" if prediction == 1 else "Likely to Stay"
            else:
                result_text = "Likely to Purchase" if prediction == 1 else "Likely to Churn"
        
        # For UI coloring: is this a "good" outcome?
        is_positive = (prediction == 0 if is_churn_inverted else prediction == 1)
        
        return {
            "prediction": int(prediction),
            "result_text": result_text,
            "probability": prob_percent,
            "is_positive": is_positive,
            "is_churn_inverted": is_churn_inverted,
            "model_source": "Personalized Business Data"
        }
    except Exception as e:
        import traceback
        logging.error(f"Churn Predict Traceback: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/report/summary")
def get_report_summary(request: Request, source: str = Query("sales")):
    username = request.session.get("username")
    df = get_report_dataframe(username, source)
    
    if df is None or df.empty:
        return {"has_data": False, "message": f"No {source} data found for your account. Please upload or generate data first."}

    try:
        data = {"has_data": True, "kpis": {}, "charts": {}, "insights": []}
        
        if source == "sales":
            # Sales KPIs
            total_sales = float(df["Sales"].sum()) if "Sales" in df.columns else (float(df["revenue"].sum()) if "revenue" in df.columns else 0.0)
            total_qty = int(df["Quantity"].sum()) if "Quantity" in df.columns else (int(df["quantity"].sum()) if "quantity" in df.columns else 0)
            total_orders = int(len(df))
            aov = total_sales / total_orders if total_orders > 0 else 0.0
            
            data["kpis"] = {
                "Total Revenue": f"{total_sales:,.2f}",
                "Total Orders": f"{total_orders:,}",
                "Units Sold": f"{total_qty:,}",
                "Avg Order Value": f"{aov:,.2f}"
            }
            
            # Trend Chart
            if ("Order Date" in df.columns or "order_date" in df.columns) and ("Sales" in df.columns or "revenue" in df.columns):
                date_col = "Order Date" if "Order Date" in df.columns else "order_date"
                val_col = "Sales" if "Sales" in df.columns else "revenue"
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df_trend = df.dropna(subset=[date_col]).copy()
                df_trend["YM"] = df_trend[date_col].dt.strftime("%Y-%m")
                trend = df_trend.groupby("YM")[val_col].sum().tail(12)
                data["charts"]["trend"] = {"labels": trend.index.tolist(), "data": [float(x) for x in trend.values.tolist()], "title": "Historical Revenue Trend", "label": "Monthly Revenue"}
                
                order_trend = df_trend.groupby("YM").size().tail(12)
                data["charts"]["order_trend"] = {"labels": order_trend.index.tolist(), "data": order_trend.values.tolist(), "title": "Monthly Order Count", "label": "Orders"}
            
            # Categories Chart
            if "Category" in df.columns and "Sales" in df.columns:
                cat = df.groupby("Category")["Sales"].sum().sort_values(ascending=False).head(5)
                data["charts"]["donut"] = {"labels": cat.index.astype(str).tolist(), "data": cat.values.tolist(), "title": "Sales by Category"}
                
            # Region Chart
            if "Region" in df.columns and "Sales" in df.columns:
                reg = df.groupby("Region")["Sales"].sum().sort_values(ascending=False).head(5)
                data["charts"]["bar"] = {"labels": reg.index.astype(str).tolist(), "data": reg.values.tolist(), "title": "Sales by Region"}

            # Loyalty Revenue Comparison
            val_col = "Sales" if "Sales" in df.columns else "revenue"
            loy_col = "LoyaltyProgram" if "LoyaltyProgram" in df.columns else ("loyalty_program" if "loyalty_program" in df.columns else None)
            if loy_col and val_col in df.columns:
                loy = df.groupby(loy_col)[val_col].sum()
                data["charts"]["loyalty"] = {
                    "labels": ["Non-Members", "Loyalty Members"],
                    "data": [float(loy.get(0, 0)), float(loy.get(1, 0))],
                    "title": "Revenue by Loyalty Segment"
                }

            data["insights"] = generate_insights(data, source="sales")

        elif source == "churn":
            # Target Column
            tc = "PurchaseStatus" if "PurchaseStatus" in df.columns else ("churn" if "churn" in df.columns else None)
            
            total_customers = int(len(df))
            churn_rate = float(df[tc].value_counts(normalize=True).get(0, 0) * 100) if tc else 0.0
            avg_satisfaction = float(df["CustomerSatisfaction"].mean()) if "CustomerSatisfaction" in df.columns else 0.0
            avg_tenure = float(df["CustomerTenureYears"].mean()) if "CustomerTenureYears" in df.columns else 0.0
            
            data["kpis"] = {
                "Total Customers": f"{total_customers:,}",
                "Predicted Churn Rate": f"{churn_rate:.1f}%",
                "Avg Satisfaction": f"{avg_satisfaction:.1f} / 5",
                "Avg Tenure (Years)": f"{avg_tenure:.1f}"
            }
            
            # Satisfaction Distribution (Bar)
            if "CustomerSatisfaction" in df.columns:
                sat = df["CustomerSatisfaction"].value_counts().sort_index()
                data["charts"]["bar"] = {"labels": sat.index.tolist(), "data": sat.values.tolist(), "title": "Satisfaction Distribution"}
                
            # Loyalty vs Churn (Donut)
            if "LoyaltyProgram" in df.columns:
                loyalty = df["LoyaltyProgram"].value_counts()
                labels = ["No Loyalty", "Loyalty Program"]
                data["charts"]["donut"] = {"labels": labels, "data": loyalty.values.tolist(), "title": "Loyalty Program Enrollment"}
                
            data["insights"] = generate_insights(data, source="churn")

        elif source == "report":
            # Same as sales but with different column defaults
            total_sales = float(df["revenue"].sum()) if "revenue" in df.columns else 0.0
            total_qty = int(df["quantity"].sum()) if "quantity" in df.columns else 0
            total_orders = int(len(df))
            aov = total_sales / total_orders if total_orders > 0 else 0.0

            data["kpis"] = {
                "Total Revenue": f"{total_sales:,.2f}",
                "Total Orders": f"{total_orders:,}",
                "Units Sold": f"{total_qty:,}",
                "Avg Order Value": f"{aov:,.2f}"
            }

            # Trend
            if "year" in df.columns and "month" in df.columns and "revenue" in df.columns:
                trend_df = df.groupby(["year", "month"])["revenue"].sum().reset_index()
                trend_df = trend_df.sort_values(by=["year", "month"])
                labels = trend_df.apply(lambda x: f"{int(x['year'])}-{int(x['month']):02d}", axis=1).tolist()
                sales_vals = [float(x) for x in trend_df["revenue"].tolist()]
                data["charts"]["trend"] = {"labels": labels, "data": sales_vals, "title": "Historical Revenue Trend", "label": "Monthly Revenue"}

            if "size" in df.columns and "revenue" in df.columns:
                sz = df.groupby("size")["revenue"].sum().sort_values(ascending=False).head(5)
                data["charts"]["donut"] = {"labels": sz.index.astype(str).tolist(), "data": sz.values.tolist(), "title": "Revenue by Size"}
            
            data["insights"] = generate_insights(data, source="sales")

        return data
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/report/download")
def download_automated_report(request: Request, source: str = Query("sales")):
    username = request.session.get("username")
    df = get_report_dataframe(username, source)
    summary_df = df.describe(include="all").transpose()
    summary_df.insert(0, "Feature", summary_df.index)
    csv_data = summary_df.to_csv(index=False)
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content=csv_data, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={source}_automated_report.csv"})

@app.post("/api/chat")
async def chat_with_bot_route(request: Request, payload: dict):
    username = request.session.get("username")
    user_df = get_user_dataframe(username)
    sm, qm, cm = get_user_models(username)
    try:
        message = payload.get("message", "")
        if not message: return JSONResponse({"error": "Empty message"}, status_code=400)
        chat_bot = ChatAssistant(user_df, sm, qm, cm)
        response = chat_bot.handle_message(message)
        return response
    except Exception as e: return JSONResponse({"error": str(e)}, status_code=500)
