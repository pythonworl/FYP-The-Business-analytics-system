import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Linear Models
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression

# Tree / Ensemble Models
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor
)

# Support Vector Regression
from sklearn.svm import SVR

# KNN
from sklearn.neighbors import KNeighborsRegressor


# =============================
# 1) Load data
# =============================
DATA_PATH = Path("Ecommerce_Sales_Data_2024_2025.csv")
df = pd.read_csv(DATA_PATH)


# =============================
# 2) Cleaning + base feature engineering
# =============================
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

df = df.dropna(subset=[
    "Order Date", "Sales", "Quantity", "Unit Price", "Discount",
    "Category", "Sub-Category", "Region", "City"
]).copy()

df["Order_Month"] = df["Order Date"].dt.month.astype(int)
df["Order_Quarter"] = df["Order Date"].dt.quarter.astype(int)
df["Order_Year"] = df["Order Date"].dt.year.astype(int)

drop_cols = ["Order ID", "Customer Name", "Product Name", "Payment Mode", "Profit", "Order Date"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])


# =============================
# 3) Helpers
# =============================
def build_preprocessor(cat_cols, num_cols):
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

def manual_metrics(y_true, preds):
    mae = mean_absolute_error(y_true, preds)
    mse = mean_squared_error(y_true, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, preds)
    return mae, rmse, r2

def evaluate_models(X, y, preprocessor, models_dict, task_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rows = []
    for name, model in models_dict.items():
        print(f"Training {task_name} with {name}...")
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mae, rmse, r2 = manual_metrics(y_test, preds)
        rows.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

    results = pd.DataFrame(rows).sort_values(["MAE", "RMSE"], ascending=True).reset_index(drop=True)
    best_name = results.loc[0, "Model"]

    print("\n==============================")
    print(f"{task_name} - Model Comparison")
    print(results)
    print(f"Best model for {task_name}: {best_name}")
    print("==============================\n")

    return results, best_name


# ==========================================================
# 4) Quantity Model (Demand) - AGGREGATED
# ==========================================================
group_cols = ["Order_Year", "Order_Month", "Category", "Sub-Category", "Region"]

qty_agg = df.groupby(group_cols, as_index=False).agg(
    Total_Quantity=("Quantity", "sum"),
    Avg_UnitPrice=("Unit Price", "mean"),
    Avg_Discount=("Discount", "mean"),
    Orders_Count=("Quantity", "count")
)

qty_cat_cols = ["Category", "Sub-Category", "Region"]
qty_num_cols = ["Order_Year", "Order_Month", "Avg_UnitPrice", "Avg_Discount", "Orders_Count"]

X_qty = qty_agg[qty_cat_cols + qty_num_cols]
y_qty = qty_agg["Total_Quantity"]

prep_qty = build_preprocessor(qty_cat_cols, qty_num_cols)

# 🔥 MANY ALGORITHMS FOR DEMAND
qty_models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(random_state=42),
    "Lasso": Lasso(random_state=42, max_iter=20000),
    "ElasticNet": ElasticNet(random_state=42, max_iter=20000),

    "RandomForest": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=400, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),

    "KNN": KNeighborsRegressor(n_neighbors=7),
    "SVR_RBF": SVR(kernel="rbf", C=10, gamma="scale"),
}

qty_results, best_qty = evaluate_models(
    X_qty, y_qty, prep_qty, qty_models, "Quantity (Aggregated Demand)"
)

best_qty_pipe = Pipeline([("prep", prep_qty), ("model", qty_models[best_qty])])
best_qty_pipe.fit(X_qty, y_qty)

joblib.dump(best_qty_pipe, "best_quantity_model.pkl")
qty_results.to_csv("quantity_model_comparison.csv", index=False)


# ===========================================
# 5) Sales Model (Revenue) - PER ORDER
# ===========================================
sales_cat_cols = ["Category", "Sub-Category", "Region", "City"]
sales_num_cols = ["Unit Price", "Discount", "Order_Month", "Order_Quarter", "Order_Year", "Quantity"]

X_sales = df[sales_cat_cols + sales_num_cols]
y_sales = df["Sales"]

prep_sales = build_preprocessor(sales_cat_cols, sales_num_cols)

# 🔥 MANY ALGORITHMS FOR SALES
sales_models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(random_state=42),
    "Lasso": Lasso(random_state=42, max_iter=20000),
    "ElasticNet": ElasticNet(random_state=42, max_iter=20000),

    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),

    "KNN": KNeighborsRegressor(n_neighbors=7),
    "SVR_RBF": SVR(kernel="rbf", C=10, gamma="scale"),
}

sales_results, best_sales = evaluate_models(
    X_sales, y_sales, prep_sales, sales_models, "Sales (Revenue)"
)

best_sales_pipe = Pipeline([("prep", prep_sales), ("model", sales_models[best_sales])])
best_sales_pipe.fit(X_sales, y_sales)

joblib.dump(best_sales_pipe, "best_sales_model.pkl")
sales_results.to_csv("sales_model_comparison.csv", index=False)


# =============================
# 6) Final message
# =============================
print("Training complete!")
print("Saved files:")
print(" - best_quantity_model.pkl")
print(" - best_sales_model.pkl")
print(" - quantity_model_comparison.csv")
print(" - sales_model_comparison.csv")
