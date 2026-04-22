import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging
import random
import json
from datetime import datetime, timedelta
from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score

# Fast Regressors
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Fast Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

APP_DIR = Path(__file__).parent
USER_DATA_DIR = APP_DIR / "user_data"

def get_user_model_dir(username: str) -> Path:
    d = USER_DATA_DIR / username / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_user_data_dir(username: str) -> Path:
    d = USER_DATA_DIR / username / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d

# ==========================================================
# Helpers
# ==========================================================
def fuzzy_column_mapper(df: pd.DataFrame, target_schema: list):
    """
    Attempts to map existing columns in df to a target schema using fuzzy matching.
    target_schema example: ["Order Date", "Sales", "Quantity"]
    """
    mapping = {
        "Order Date": ["date", "order_date", "transaction_date", "timestamp", "dt", "year", "yr", "y"],
        "Sales": ["sales", "revenue", "amount", "total_sales", "price_total"],
        "Quantity": ["quantity", "qty", "units", "count", "number_of_items", "volume", "sales volume"],
        "Unit Price": ["unit_price", "price", "rate", "cost_per_unit", "base price"],
        "Sub-Category": ["sub_category", "sub-category", "subcat", "type", "fuel type"],
        "Category": ["category", "cat", "group", "class", "data", "uploaded_data", "model"],
        "Region": ["region", "reg", "area", "location", "zone"],
        "City": ["city", "town", "loc"],
        "PurchaseStatus": ["purchasestatus", "churn", "status", "target", "label", "outcome"]
    }
    
    new_cols = {}
    for official_name in target_schema:
        if official_name in df.columns:
            continue # already exact match
            
        # Try variations
        variations = mapping.get(official_name, [])
        found = False
        for v in variations:
            # Check case-insensitive and stripped
            # AND skip if the source column is already mapped
            matches = [c for c in df.columns if c.strip().lower() == v.lower() and c not in new_cols]
            if matches:
                new_cols[matches[0]] = official_name
                found = True
                break
        
        if not found:
            # Last resort: check if any official name or variation is in the column name (substring match)
            # AND skip if the source column is already mapped
            all_alternatives = [official_name.lower()] + [v.lower() for v in variations]
            for alt in all_alternatives:
                matches = [c for c in df.columns if alt in c.strip().lower() and c not in new_cols]
                if matches:
                    new_cols[matches[0]] = official_name
                    found = True
                    break
                 
    return df.rename(columns=new_cols)

def build_reg_preprocessor(cat_cols, num_cols):
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

def evaluate_reg_models(X, y, preprocessor, models_dict):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_name = None
    best_mae = float("inf")
    best_pipe = None
    all_metrics = {}

    for name, model in models_dict.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        all_metrics[name] = round(float(mae), 4)
        
        logging.info(f"[Regression] Model: {name} - MAE: {mae:.2f}")
        
        if mae < best_mae:
            best_mae = mae
            best_name = name
            best_pipe = pipe

    logging.info(f"Best Regression Model Selected: {best_name} with MAE: {best_mae:.2f}")
    return best_name, best_pipe, all_metrics

def evaluate_clf_models(X, y, preprocessor, models_dict):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_name = None
    best_f1 = -1
    best_pipe = None
    all_metrics = {}

    for name, model in models_dict.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        all_metrics[name] = round(float(f1), 4)
        
        logging.info(f"[Classification] Model: {name} - F1-Score: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_pipe = pipe

    logging.info(f"Best Classification Model Selected: {best_name} with F1: {best_f1:.4f}")
    return best_name, best_pipe, all_metrics

# ==========================================================
# Sample Data Generation
# ==========================================================
def get_industry_config(category_name: str):
    """
    Returns realistic pricing, quantity, and sub-category ranges based on industry type.
    """
    name = category_name.lower()
    
    # Industries
    if any(k in name for k in ["ice cream", "dessert", "food"]):
        return {
            "price_range": (2.5, 12.0),
            "qty_range": (1, 10),
            "subcats": ["Vanilla", "Chocolate", "Strawberry", "Mango", "Pistachio", "Cookies & Cream"]
        }
    elif any(k in name for k in ["car", "auto", "vehicle", "truck"]):
        return {
            "price_range": (15000.0, 60000.0),
            "qty_range": (1, 2),
            "subcats": ["Sedan", "SUV", "Hatchback", "Luxury Sport", "Crossover", "Electric"]
        }
    elif any(k in name for k in ["laptop", "phone", "electronic", "computer", "tech"]):
        return {
            "price_range": (300.0, 2500.0),
            "qty_range": (1, 3),
            "subcats": ["Gaming", "Business", "Ultrabook", "Workstation", "Budget", "Peripherals"]
        }
    elif any(k in name for k in ["cloth", "shirt", "apparel", "wear"]):
        return {
            "price_range": (15.0, 150.0),
            "qty_range": (1, 5),
            "subcats": ["Casual Wear", "Formal Wear", "Sportswear", "Outerwear", "Accessories"]
        }
    elif any(k in name for k in ["furniture", "home", "decor", "office"]):
        return {
            "price_range": (50.0, 3000.0),
            "qty_range": (1, 3),
            "subcats": ["Living Room", "Bedroom", "Office Space", "Kitchen", "Outdoor"]
        }
    
    # Default range for unknown
    return {
        "price_range": (50.0, 500.0),
        "qty_range": (1, 15),
        "subcats": ["Standard Edition", "Premium Pack", "Basic Version", "Pro Series"]
    }

def generate_sample_csv(username: str, category_name: str, data_type: str = "sales", record_count: int = 500):
    """
    Generates a synthetic CSV file for the user.
    """
    data_dir = get_user_data_dir(username)
    
    if data_type == "sales":
        config = get_industry_config(category_name)
        start_date = datetime(2023, 1, 1)
        data = []
        
        regions = ["North", "South", "East", "West", "Central"]
        cities = ["New York", "London", "Tokyo", "Paris", "Berlin", "Dubai", "Singapore"]
        
        for _ in range(record_count):
            order_date = start_date + timedelta(days=random.randint(0, 730))
            sub = random.choice(config["subcats"])
            reg = random.choice(regions)
            cit = random.choice(cities)
            
            # Use contextual configs
            price = round(random.uniform(config["price_range"][0], config["price_range"][1]), 2)
            qty = random.randint(config["qty_range"][0], config["qty_range"][1])
            
            disc = random.choice([0, 5, 10, 15, 20])
            sales = round(price * qty * (1 - disc/100), 2)
            
            data.append({
                "Order Date": order_date.strftime("%Y-%m-%d"),
                "Category": category_name,
                "Sub-Category": sub,
                "Region": reg,
                "City": cit,
                "Unit Price": price,
                "Quantity": qty,
                "Discount": disc,
                "Sales": sales
            })
            
        df = pd.DataFrame(data)
        file_path = data_dir / "sales_data.csv"
        df.to_csv(file_path, index=False)
        return str(file_path)

    elif data_type == "churn":
        data = []
        for _ in range(record_count):
            # Realistic Behavioral Data
            n_purchases = random.randint(1, 100)
            tenure = round(random.uniform(0.5, 10), 1)
            last_p = random.randint(1, 365)
            sat = random.randint(1, 5)
            disc = random.randint(0, 10)
            is_loyalty = random.choice([0, 1])
            
            # Loyalty Scoring Logic
            # Base logic: Positive behaviors increase the "Retention Score"
            # Threshold chosen to create ~50/50 split but with real correlations
            score = (sat * 20) + (n_purchases * 0.8) + (tenure * 10) + (is_loyalty * 30) + (disc * 5) - (last_p * 0.25)
            noise = random.randint(-40, 40)
            
            # 1 = Stay/Purchase, 0 = Churn
            purchase_status = 1 if (score + noise) > 75 else 0
            
            data.append({
                "NumberOfPurchases": n_purchases,
                "CustomerTenureYears": tenure,
                "LastPurchaseDaysAgo": last_p,
                "CustomerSatisfaction": sat,
                "DiscountsAvailed": disc,
                "LoyaltyProgram": is_loyalty,
                "PurchaseStatus": purchase_status
            })
        df = pd.DataFrame(data)
        file_path = data_dir / "churn_data.csv"
        df.to_csv(file_path, index=False)
        return str(file_path)

# ==========================================================
# Training Functions
# ==========================================================

def train_user_sales_and_quantity_models(username: str, df: pd.DataFrame):
    model_dir = get_user_model_dir(username)
    
    # Robust Category Cleaning
    if "Category" in df.columns:
        df["Category"] = df["Category"].astype(str).str.replace(r'[^\x00-\x7F]+', 'a', regex=True).str.strip()
        df["Category"] = df["Category"].apply(lambda x: "".join([c for c in x if ord(c) < 128]))

    # Fuzzy Mapping for Industry Terms
    target_col_search = ["Order Date", "Category", "Sub-Category", "Region", "City", "Sales", "Quantity", "Unit Price", "Discount", "product_type", "units_sold", "revenue", "Date"]
    df = fuzzy_column_mapper(df, target_col_search)

    # ============================================================
    # SMART TARGET DETECTION FOR SALES
    # Find the sales/revenue target column dynamically
    # ============================================================
    sales_target_col = None
    for col in df.columns:
        col_lower = col.strip().lower()
        if col_lower in ['sales', 'revenue', 'amount', 'total_sales', 'price_total', 'total_revenue']:
            sales_target_col = col
            break
    
    if sales_target_col is None:
        # Try to compute Sales from Unit Price * Quantity
        if "Unit Price" in df.columns and "Quantity" in df.columns:
            df["Sales"] = df["Unit Price"] * df["Quantity"]
            sales_target_col = "Sales"
        else:
            return {"status": "error", "message": "No sales/revenue target column found in data."}
    
    original_target_name = sales_target_col
    logging.info(f"Detected sales target column: '{sales_target_col}'")

    # Ensure target is numeric
    df[sales_target_col] = pd.to_numeric(df[sales_target_col], errors="coerce")
    df = df.dropna(subset=[sales_target_col]).copy()
    
    if df.empty:
        return {"status": "error", "message": "No valid data left after cleaning."}

    # ============================================================
    # DATE FEATURE EXTRACTION (if date column exists)
    # ============================================================
    has_date = False
    if "Order Date" in df.columns:
        def robust_year_detect(val):
            try:
                if isinstance(val, (int, float)) and 1900 < val < 2100: return int(val)
                s = str(val).strip()
                if s.isdigit() and 1900 < int(s) < 2100: return int(s)
                dt = pd.to_datetime(val, errors="coerce")
                return dt.year if pd.notnull(dt) else 2025
            except: return 2025
        
        df["Order_Year"] = df["Order Date"].apply(robust_year_detect)
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
        df["Order_Month"] = df["Order Date"].dt.month.fillna(1).astype(int)
        df["Order_Quarter"] = df["Order Date"].dt.quarter.fillna(1).astype(int)
        has_date = True

    # ============================================================
    # DYNAMIC FEATURE DETECTION FOR SALES
    # Same pattern as churn: detect features from the data
    # ============================================================
    
    # Start with all columns except the target and non-predictive ones
    X = df.drop(columns=[sales_target_col], errors='ignore')
    
    # Drop columns that are clearly not predictive
    drop_patterns = ['id', 'name', 'email', 'phone', 'address', 'timestamp', 'index',
                     'order date', 'date', 'profit', 'order_date']
    cols_to_drop = [c for c in X.columns if c.strip().lower() in drop_patterns 
                    or any(p == c.strip().lower() for p in drop_patterns)]
    # Keep derived date features (Order_Year, Order_Month, Order_Quarter)
    cols_to_drop = [c for c in cols_to_drop if c not in ['Order_Year', 'Order_Month', 'Order_Quarter']]
    if cols_to_drop:
        logging.info(f"Sales: Dropping non-predictive columns: {cols_to_drop}")
        X = X.drop(columns=cols_to_drop, errors='ignore')
    
    if X.empty or X.shape[1] == 0:
        return {"status": "error", "message": "No usable feature columns found in sales data after filtering."}
    
    # Identify numeric vs categorical columns dynamically
    numerical_features = []
    categorical_features = []
    
    for col in X.columns:
        # Try to convert to numeric
        numeric_check = pd.to_numeric(X[col], errors='coerce')
        non_null_ratio = numeric_check.notna().sum() / len(X) if len(X) > 0 else 0
        
        if non_null_ratio >= 0.5:  # At least 50% of values are numeric
            numerical_features.append(col)
            X[col] = numeric_check  # Convert to numeric
        else:
            n_unique = X[col].nunique()
            if n_unique <= 50:  # Reasonable number of categories
                categorical_features.append(col)
            else:
                logging.info(f"Sales: Skipping high-cardinality column: {col} ({n_unique} unique values)")
    
    all_features = numerical_features + categorical_features
    feature_order = [f for f in X.columns if f in all_features]
    
    if not all_features:
        return {"status": "error", "message": "No valid numeric or categorical features detected in sales data."}
    
    # Keep only detected features
    X = X[all_features]
    y_sales = df[sales_target_col]
    
    logging.info(f"Sales Dynamic Feature Detection: {len(numerical_features)} numeric, {len(categorical_features)} categorical")
    logging.info(f"  Numeric: {numerical_features}")
    logging.info(f"  Categorical: {categorical_features}")

    # ============================================================
    # 1) QUANTITY MODEL (only if Quantity column exists)
    # ============================================================
    best_qty_name = "None"
    qty_metrics = {}
    
    qty_target_candidates = [c for c in df.columns if c.strip().lower() in ['quantity', 'qty', 'units', 'units_sold', 'count']]
    
    if qty_target_candidates and has_date:
        qty_target = qty_target_candidates[0]
        df[qty_target] = pd.to_numeric(df[qty_target], errors="coerce")
        
        # Build quantity-specific features from what's available
        qty_cat_cols = [c for c in categorical_features if c in df.columns]
        qty_group_cols = ["Order_Year", "Order_Month"] + qty_cat_cols
        available_group_cols = [c for c in qty_group_cols if c in df.columns]
        
        if available_group_cols:
            agg_dict = {qty_target: [('Total_Quantity', 'sum'), ('Orders_Count', 'count')]}
            # Add numeric averaging for available numeric features
            num_agg = {}
            for nc in numerical_features:
                if nc not in ['Order_Year', 'Order_Month', 'Order_Quarter', qty_target] and nc in df.columns:
                    num_agg[nc] = [(f'Avg_{nc}', 'mean')]
            
            try:
                qty_agg = df.groupby(available_group_cols, as_index=False).agg(
                    Total_Quantity=(qty_target, "sum"),
                    Orders_Count=(qty_target, "count")
                )
                
                # Add averaged numeric columns
                for nc in numerical_features:
                    if nc not in available_group_cols and nc != qty_target and nc in df.columns:
                        avg_vals = df.groupby(available_group_cols, as_index=False)[nc].mean()
                        qty_agg = qty_agg.merge(avg_vals, on=available_group_cols, how='left')
                        qty_agg = qty_agg.rename(columns={nc: f'Avg_{nc}'})
                
                qty_num_for_model = [c for c in qty_agg.columns if c.startswith('Avg_') or c in ['Order_Year', 'Order_Month']]
                qty_cat_for_model = [c for c in qty_cat_cols if c in qty_agg.columns]
                
                X_qty = qty_agg[qty_cat_for_model + qty_num_for_model]
                y_qty = qty_agg["Total_Quantity"]
                
                prep_qty = build_reg_preprocessor(qty_cat_for_model, qty_num_for_model)
                
                reg_models = {
                    "LinearRegression": LinearRegression(),
                    "Ridge": Ridge(random_state=42),
                    "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=42),
                    "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
                }
                
                logging.info(f"Training Quantity Model for user {username}...")
                best_qty_name, best_qty_pipe, qty_metrics = evaluate_reg_models(X_qty, y_qty, prep_qty, reg_models)
                best_qty_pipe.fit(X_qty, y_qty)
                joblib.dump(best_qty_pipe, model_dir / "best_quantity_model.pkl")
            except Exception as e:
                logging.error(f"Quantity Training Failed for {username}: {e}")
                best_qty_name, qty_metrics = "None", {"Error": str(e)}
    else:
        logging.info(f"No quantity target or date column found for {username}, skipping quantity model.")

    # ============================================================
    # 2) SALES MODEL (Dynamic Features)
    # ============================================================
    
    # Build dynamic preprocessor
    transformers = []
    if numerical_features:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, numerical_features))
    
    if categorical_features:
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_transformer, categorical_features))
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    
    reg_models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=42),
        "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    }

    logging.info(f"Training Sales Model for user {username} with {len(all_features)} dynamic features...")
    try:
        best_sales_name, best_sales_pipe, sales_metrics = evaluate_reg_models(X, y_sales, preprocessor, reg_models)
        # Retrain on full data
        best_sales_pipe.fit(X, y_sales)
        joblib.dump(best_sales_pipe, model_dir / "best_sales_model.pkl")
    except Exception as e:
        logging.error(f"Sales Training Failed for {username}: {e}")
        best_sales_name, sales_metrics = "None", {"Error": str(e)}

    # Save feature config for prediction alignment
    feature_config = {
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "all_features": all_features,
        "feature_order": feature_order,
        "original_target_name": original_target_name
    }
    with open(model_dir / "sales_feature_config.json", "w") as f:
        json.dump(feature_config, f, indent=2)

    # Save sales feature metadata for dynamic UI
    try:
        analyze_and_save_sales_feature_metadata(username, df, categorical_features, numerical_features, X, y_sales, original_target_name)
    except Exception as e:
        logging.error(f"Sales metadata analysis failed: {e}")

    return {
        "status": "success", 
        "qty_model": best_qty_name, "qty_metrics": qty_metrics,
        "sales_model": best_sales_name, "sales_metrics": sales_metrics
    }


def analyze_and_save_sales_feature_metadata(username: str, df: pd.DataFrame,
                                             sales_cat_cols: list, sales_num_cols: list,
                                             X_sales: pd.DataFrame, y_sales: pd.Series,
                                             original_target_name: str = "Sales"):
    """
    Saves sales feature metadata for the frontend to render dynamic forms.
    Uses Random Forest feature importance for ranking.
    """
    model_dir = get_user_model_dir(username)
    
    try:
        from sklearn.ensemble import RandomForestRegressor as RFR
        
        transformers = []
        if sales_num_cols:
            num_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', num_pipe, sales_num_cols))
        if sales_cat_cols:
            cat_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', cat_pipe, sales_cat_cols))
        
        if not transformers:
            return
        
        prep = ColumnTransformer(transformers=transformers, remainder='drop')
        rf = RFR(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        pipe = Pipeline([('prep', prep), ('rf', rf)])
        pipe.fit(X_sales, y_sales)
        
        importances = pipe.named_steps['rf'].feature_importances_
        
        # Map importances back to original features
        transformed_names = list(sales_num_cols)
        if sales_cat_cols:
            ohe = prep.named_transformers_['cat'].named_steps['encoder']
            cat_names = ohe.get_feature_names_out(sales_cat_cols).tolist()
            transformed_names.extend(cat_names)
        
        feature_importance_map = {}
        for fname, imp in zip(transformed_names, importances):
            base_name = None
            for cat_col in sales_cat_cols:
                if fname.startswith(cat_col + '_'):
                    base_name = cat_col
                    break
            key = base_name if base_name else fname
            feature_importance_map[key] = feature_importance_map.get(key, 0) + imp
        
        features_meta = []
        all_cols = sales_num_cols + sales_cat_cols
        for feat_name in all_cols:
            imp_score = feature_importance_map.get(feat_name, 0.0)
            
            if feat_name in sales_num_cols:
                col_data = pd.to_numeric(X_sales[feat_name], errors='coerce').dropna()
                features_meta.append({
                    "name": feat_name,
                    "type": "numeric",
                    "importance": round(float(imp_score), 4),
                    "min": round(float(col_data.min()), 2) if len(col_data) > 0 else 0,
                    "max": round(float(col_data.max()), 2) if len(col_data) > 0 else 100,
                    "median": round(float(col_data.median()), 2) if len(col_data) > 0 else 0,
                    "mean": round(float(col_data.mean()), 2) if len(col_data) > 0 else 0
                })
            elif feat_name in sales_cat_cols:
                unique_vals = X_sales[feat_name].dropna().unique().tolist()
                unique_vals = [str(v) for v in unique_vals][:100]
                features_meta.append({
                    "name": feat_name,
                    "type": "categorical",
                    "importance": round(float(imp_score), 4),
                    "values": sorted(unique_vals)
                })
        
        features_meta.sort(key=lambda x: x["importance"], reverse=True)
        
        metadata = {
            "features": features_meta,
            "target_column": original_target_name,
            "original_target_name": original_target_name,
            "total_features": len(features_meta),
            "total_samples": int(len(X_sales)),
            "model_type": "regression",
            "numerical_features": sales_num_cols,
            "categorical_features": sales_cat_cols
        }
        
        with open(model_dir / "sales_feature_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"Sales feature metadata saved for {username}: {len(features_meta)} features")
        
    except Exception as e:
        logging.error(f"Sales feature importance analysis failed for {username}: {e}")


def analyze_and_save_feature_metadata(username: str, X: pd.DataFrame, y: pd.Series, 
                                       numerical_features: list, categorical_features: list,
                                       feature_order: list,
                                       is_churn_inverted: bool = False,
                                       original_target_name: str = "PurchaseStatus"):
    """
    Runs Random Forest feature importance on the dataset and saves
    a detailed feature metadata JSON for the frontend to render dynamic forms.
    """
    model_dir = get_user_model_dir(username)
    
    try:
        # Build a quick RF for importance analysis
        from sklearn.ensemble import RandomForestClassifier as RFC
        
        # Create a simple preprocessor for importance analysis
        transformers = []
        if numerical_features:
            num_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', num_pipe, numerical_features))
        if categorical_features:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features))
        
        if not transformers:
            return
            
        prep = ColumnTransformer(transformers=transformers, remainder='drop')
        
        rf = RFC(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        pipe = Pipeline([('prep', prep), ('rf', rf)])
        pipe.fit(X, y)
        
        # Get feature importances
        importances = pipe.named_steps['rf'].feature_importances_
        
        # Map importances back to original feature names
        transformed_names = []
        if numerical_features:
            transformed_names.extend(numerical_features)
        if categorical_features:
            ohe = prep.named_transformers_['cat']
            cat_names = ohe.get_feature_names_out(categorical_features).tolist()
            transformed_names.extend(cat_names)
        
        # Aggregate importance for categorical features (sum across one-hot columns)
        feature_importance_map = {}
        for fname, imp in zip(transformed_names, importances):
            # Check if this is a one-hot encoded feature
            base_name = None
            for cat_col in categorical_features:
                if fname.startswith(cat_col + '_'):
                    base_name = cat_col
                    break
            
            key = base_name if base_name else fname
            feature_importance_map[key] = feature_importance_map.get(key, 0) + imp
        
        # Build metadata for each feature
        features_meta = []
        for feat_name in feature_order:
            imp_score = feature_importance_map.get(feat_name, 0.0)
            
            if feat_name in numerical_features:
                col_data = pd.to_numeric(X[feat_name], errors='coerce').dropna()
                features_meta.append({
                    "name": feat_name,
                    "type": "numeric",
                    "importance": round(float(imp_score), 4),
                    "min": round(float(col_data.min()), 2) if len(col_data) > 0 else 0,
                    "max": round(float(col_data.max()), 2) if len(col_data) > 0 else 100,
                    "median": round(float(col_data.median()), 2) if len(col_data) > 0 else 0,
                    "mean": round(float(col_data.mean()), 2) if len(col_data) > 0 else 0
                })
            elif feat_name in categorical_features:
                unique_vals = X[feat_name].dropna().unique().tolist()
                # Convert to strings for JSON safety
                unique_vals = [str(v) for v in unique_vals][:50]  # Cap at 50 values
                features_meta.append({
                    "name": feat_name,
                    "type": "categorical",
                    "importance": round(float(imp_score), 4),
                    "values": sorted(unique_vals)
                })
        
        # Sort by importance (highest first)
        features_meta.sort(key=lambda x: x["importance"], reverse=True)
        
        metadata = {
            "features": features_meta,
            "target_column": y.name if hasattr(y, 'name') else "PurchaseStatus",
            "original_target_name": original_target_name,
            "is_churn_inverted": is_churn_inverted,
            "total_features": len(features_meta),
            "total_samples": int(len(X)),
            "model_type": "classification"
        }
        
        with open(model_dir / "churn_feature_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"Feature metadata saved for {username}: {len(features_meta)} features detected")
        
    except Exception as e:
        logging.error(f"Feature importance analysis failed for {username}: {e}")


def get_feature_metadata(username: str, model_type: str = "churn") -> dict:
    """
    Retrieves saved feature metadata for a user's model.
    Returns None if no metadata exists.
    """
    model_dir = get_user_model_dir(username)
    meta_file = model_dir / f"{model_type}_feature_metadata.json"
    
    if meta_file.exists():
        with open(meta_file, "r") as f:
            return json.load(f)
    return None


def train_user_churn_model(username: str, df: pd.DataFrame):
    model_dir = get_user_model_dir(username)
    
    # ============================================================
    # SMART TARGET DETECTION
    # Detect the original target column BEFORE fuzzy mapping
    # to determine if labels need to be inverted.
    # In a "churn" dataset: 1 = churned (bad), 0 = stayed (good)
    # In a "PurchaseStatus" dataset: 1 = purchased (good), 0 = churned (bad)
    # ============================================================
    churn_indicators = ['churn', 'churned', 'attrition', 'left', 'exited']
    purchase_indicators = ['purchasestatus', 'purchase_status', 'purchased', 'converted', 'buy']
    
    # Find the original target column name
    original_target_name = None
    for col in df.columns:
        col_lower = col.strip().lower()
        if col_lower in [c.lower() for c in churn_indicators + purchase_indicators + ['purchasestatus', 'status', 'target', 'label', 'outcome']]:
            original_target_name = col
            break
    
    # Determine if labels are "churn-style" (1=bad) vs "purchase-style" (1=good)
    is_churn_inverted = False
    if original_target_name:
        name_lower = original_target_name.strip().lower()
        if any(ci in name_lower for ci in churn_indicators):
            is_churn_inverted = True  # 1 means churned, NOT purchased
            logging.info(f"Detected CHURN-style target column: '{original_target_name}' (1=churned, 0=stayed)")
        else:
            logging.info(f"Detected PURCHASE-style target column: '{original_target_name}' (1=purchased, 0=churned)")
    else:
        logging.info("No known target column found, will use fuzzy mapping")
    
    # Fuzzy Mapping 
    target_col_search = ["PurchaseStatus"]
    df = fuzzy_column_mapper(df, target_col_search)
    target_col = "PurchaseStatus"
    
    if target_col not in df.columns:
        return {"status": "error", "message": f"'{target_col}' column missing or could not be mapped from data"}

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ============================================================
    # DYNAMIC FEATURE DETECTION
    # Instead of hardcoding features, detect them from the data.
    # ============================================================
    
    # Drop columns that are clearly not predictive (IDs, names, etc.)
    drop_patterns = ['id', 'name', 'email', 'phone', 'address', 'date', 'timestamp', 'index']
    cols_to_drop = [c for c in X.columns if any(p in c.strip().lower() for p in drop_patterns)]
    if cols_to_drop:
        logging.info(f"Dropping non-predictive columns: {cols_to_drop}")
        X = X.drop(columns=cols_to_drop, errors='ignore')
    
    if X.empty or X.shape[1] == 0:
        return {"status": "error", "message": "No usable feature columns found in churn data after filtering."}
    
    # Identify numeric vs categorical columns dynamically
    numerical_features = []
    categorical_features = []
    
    for col in X.columns:
        # Try to convert to numeric
        numeric_check = pd.to_numeric(X[col], errors='coerce')
        non_null_ratio = numeric_check.notna().sum() / len(X) if len(X) > 0 else 0
        
        if non_null_ratio >= 0.5:  # At least 50% of values are numeric
            numerical_features.append(col)
            X[col] = numeric_check  # Convert to numeric
        else:
            n_unique = X[col].nunique()
            if n_unique <= 50:  # Reasonable number of categories
                categorical_features.append(col)
            else:
                logging.info(f"Skipping high-cardinality column: {col} ({n_unique} unique values)")
    
    all_features = numerical_features + categorical_features
    feature_order = list(X.columns)  # Preserve original column order for metadata
    feature_order = [f for f in feature_order if f in all_features]
    
    if not all_features:
        return {"status": "error", "message": "No valid numeric or categorical features detected in churn data."}
    
    # Keep only detected features
    X = X[all_features]
    
    logging.info(f"Dynamic Feature Detection: {len(numerical_features)} numeric, {len(categorical_features)} categorical")
    logging.info(f"  Numeric: {numerical_features}")
    logging.info(f"  Categorical: {categorical_features}")

    if X.isnull().all().all():
         return {"status": "error", "message": "No valid data found for churn prediction features."}

    # Build dynamic preprocessor
    transformers = []
    if numerical_features:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, numerical_features))
    
    if categorical_features:
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    clf_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(max_depth=15, min_samples_leaf=3, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=3, random_state=42, n_jobs=-1, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
    }

    logging.info(f"Training Churn Model for user {username}...")
    best_clf_name, best_clf_pipe, churn_metrics = evaluate_clf_models(X, y, preprocessor, clf_models)

    # Retrain on full data
    best_clf_pipe.fit(X, y)
    joblib.dump(best_clf_pipe, model_dir / "best_churn_model.pkl")
    
    # Save the feature list used for training (critical for prediction alignment)
    feature_config = {
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "all_features": all_features,
        "feature_order": feature_order,
        "is_churn_inverted": is_churn_inverted,
        "original_target_name": original_target_name or target_col
    }
    with open(model_dir / "churn_feature_config.json", "w") as f:
        json.dump(feature_config, f, indent=2)
    
    # Run Feature Importance Analysis and save metadata for the UI
    analyze_and_save_feature_metadata(username, X, y, numerical_features, categorical_features, feature_order,
                                       is_churn_inverted=is_churn_inverted,
                                       original_target_name=original_target_name or target_col)

    return {"status": "success", "churn_model": best_clf_name, "churn_metrics": churn_metrics}

# ==========================================================
# Batch Predictions Export
# ==========================================================
def generate_batch_predictions(username: str):
    """
    Generates independent CSVs of predictions for both Sales and Churn data.
    Saves them as:
    - sales_batch_predictions.csv
    - churn_batch_predictions.csv
    """
    data_dir = get_user_data_dir(username)
    model_dir = get_user_model_dir(username)
    
    # --- 1. SALES BATCH PREDICTIONS ---
    sales_file = data_dir / "sales_data.csv"
    sm_path = model_dir / "best_sales_model.pkl"
    sales_config_path = model_dir / "sales_feature_config.json"
    
    if sales_file.exists() and sm_path.exists():
        try:
            import json as json_mod
            df_s = pd.read_csv(sales_file)
            sm = joblib.load(sm_path)
            
            # Load dynamic feature config
            if sales_config_path.exists():
                with open(sales_config_path, "r") as f:
                    feat_config = json_mod.load(f)
                numerical_features = feat_config.get("numerical_features", [])
                categorical_features = feat_config.get("categorical_features", [])
                all_features = feat_config.get("all_features", numerical_features + categorical_features)
            else:
                # Legacy fallback
                all_features = ["Unit Price", "Discount", "Order_Month", "Order_Quarter", "Order_Year", "Quantity", "Category", "Sub-Category", "Region"]
                numerical_features = ["Unit Price", "Discount", "Order_Month", "Order_Quarter", "Order_Year", "Quantity"]
                categorical_features = ["Category", "Sub-Category", "Region"]
            
            # Fuzzy-map columns
            df_s = fuzzy_column_mapper(df_s, all_features + ["Order Date", "Sales"])
            
            # Generate date-derived features if needed
            if "Order Date" in df_s.columns:
                df_s["Order Date"] = pd.to_datetime(df_s["Order Date"], errors="coerce")
                if "Order_Month" in all_features:
                    df_s["Order_Month"] = df_s["Order Date"].dt.month.fillna(1).astype(int)
                if "Order_Quarter" in all_features:
                    df_s["Order_Quarter"] = df_s["Order Date"].dt.quarter.fillna(1).astype(int)
                if "Order_Year" in all_features:
                    df_s["Order_Year"] = df_s["Order Date"].dt.year.fillna(2025).astype(int)
            
            # Build X_sales with the correct columns
            X_sales = df_s.copy()
            for col in all_features:
                if col not in X_sales.columns:
                    X_sales[col] = np.nan
            
            # Convert numeric columns
            for col in numerical_features:
                if col in X_sales.columns:
                    X_sales[col] = pd.to_numeric(X_sales[col], errors='coerce')
            
            X_sales = X_sales[all_features]
            
            df_s["Predicted_Sales"] = sm.predict(X_sales)
            df_s.to_csv(data_dir / "sales_batch_predictions.csv", index=False)
            logging.info(f"Sales batch predictions generated for {username}")
        except Exception as e:
            logging.error(f"Error generating sales batch predictions: {e}")

    # --- 2. CHURN BATCH PREDICTIONS ---
    churn_file = data_dir / "churn_data.csv"
    cm_path = model_dir / "best_churn_model.pkl"
    config_path = model_dir / "churn_feature_config.json"
    
    if churn_file.exists() and cm_path.exists():
        try:
            import json as json_mod
            df_c = pd.read_csv(churn_file)
            cm = joblib.load(cm_path)
            
            # Load dynamic feature config
            if config_path.exists():
                with open(config_path, "r") as f:
                    feat_config = json_mod.load(f)
                numerical_features = feat_config.get("numerical_features", [])
                categorical_features = feat_config.get("categorical_features", [])
                all_features = feat_config.get("all_features", numerical_features + categorical_features)
                is_churn_inverted = feat_config.get("is_churn_inverted", False)
            else:
                # Fallback for legacy models
                numerical_features = [
                    'NumberOfPurchases', 'CustomerTenureYears', 
                    'LastPurchaseDaysAgo', 'CustomerSatisfaction', 'DiscountsAvailed', 'LoyaltyProgram'
                ]
                all_features = numerical_features
                categorical_features = []
                is_churn_inverted = False
            
            # Fuzzy-map columns
            df_c = fuzzy_column_mapper(df_c, all_features + ["PurchaseStatus"])
            
            # Build X_churn with the correct columns
            X_churn = df_c.copy()
            for col in all_features:
                if col not in X_churn.columns:
                    X_churn[col] = np.nan
            
            # Convert numeric columns
            for col in numerical_features:
                if col in X_churn.columns:
                    X_churn[col] = pd.to_numeric(X_churn[col], errors='coerce')
            
            X_churn = X_churn[all_features]
            
            # Generate predictions
            df_c["Churn_Prediction"] = cm.predict(X_churn)
            if hasattr(cm, "predict_proba"):
                proba = cm.predict_proba(X_churn)
                churn_class_idx = 1 if proba.shape[1] > 1 else 0
                df_c["Churn_Probability"] = proba[:, churn_class_idx]
            
            # Apply correct label semantics
            if is_churn_inverted:
                df_c["Churn_Result"] = df_c["Churn_Prediction"].apply(
                    lambda x: "Likely to Churn" if x == 1 else "Likely to Stay")
            else:
                df_c["Churn_Result"] = df_c["Churn_Prediction"].apply(
                    lambda x: "Likely to Purchase" if x == 1 else "Likely to Churn")
            
            df_c.to_csv(data_dir / "churn_batch_predictions.csv", index=False)
            logging.info(f"Churn batch predictions generated for {username}")
        except Exception as e:
            logging.error(f"Error generating churn batch predictions: {e}")

    return True

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="AutoML Trainer for Analytics Pro")
    parser.add_argument("--username", type=str, required=True, help="User to train models for")
    
    args = parser.parse_args()
    
    # 1) Get User Data
    data_dir = get_user_data_dir(args.username)
    sales_file = data_dir / "sales_data.csv"
    churn_file = data_dir / "churn_data.csv"
    
    # 2) Train Sales/Demand
    results = {}
    if sales_file.exists():
        logging.info(f"Trainer: Loading sales data for {args.username}...")
        df_sales = pd.read_csv(sales_file)
        res_s = train_user_sales_and_quantity_models(args.username, df_sales)
        results["sales_demand"] = res_s
    
    # 3) Train Churn
    if churn_file.exists():
        logging.info(f"Trainer: Loading churn data for {args.username}...")
        df_churn = pd.read_csv(churn_file)
        res_c = train_user_churn_model(args.username, df_churn)
        results["churn"] = res_c
        
    # 4) Save Metadata
    if results:
        model_dir = get_user_model_dir(args.username)
        with open(model_dir / "model_metadata.json", "w") as f:
            json.dump(results, f)
        logging.info(f"Trainer: Metadata saved for {args.username}")
        
    logging.info("Trainer Process Completed Successfully.")
