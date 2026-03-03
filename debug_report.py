
import pandas as pd
from pathlib import Path
import sys

# Mocking the path setup from main.py
try:
    APP_DIR = Path("c:/Users/fnu.sawera/OneDrive - University of Central Asia/Desktop/Business Anlytics")
    REPORT_DATA_PATH = APP_DIR / "women_clothing_ecommerce_sales.csv"
    
    print(f"Loading data from {REPORT_DATA_PATH}")
    df_report = pd.read_csv(REPORT_DATA_PATH)
    print("Data loaded. Columns:", df_report.columns.tolist())
    
    # Ensure date column is datetime
    df_report["order_date"] = pd.to_datetime(df_report["order_date"], errors="coerce")
    df_report["year"] = df_report["order_date"].dt.year
    df_report["month"] = df_report["order_date"].dt.month
    
    print("Date conversion done.")
    
    # 1. KPIs
    total_sales = df_report["revenue"].sum()
    total_qty = df_report["quantity"].sum()
    total_orders = len(df_report)
    aov = total_sales / total_orders if total_orders > 0 else 0
    
    print("KPIs calculated.")
    
    # 2. Monthly Trend
    trend_df = df_report.groupby(["year", "month"], as_index=False)["revenue"].sum()
    trend_df = trend_df.sort_values(by=["year", "month"])
    
    trend_labels = trend_df.apply(lambda x: f"{int(x['year'])}-{int(x['month']):02d}", axis=1).tolist()
    trend_sales = trend_df["revenue"].tolist()
    
    print("Trend calculated.")

    # 3. Size Share
    size_df = df_report.groupby("size", as_index=False)["revenue"].sum().sort_values(by="revenue", ascending=False)
    size_labels = size_df["size"].astype(str).tolist()
    size_data = size_df["revenue"].tolist()
    
    print("Size share calculated.")

    # 4. Color Distribution
    color_df = df_report.groupby("color", as_index=False)["revenue"].sum().sort_values(by="revenue", ascending=False).head(10)
    color_labels = color_df["color"].astype(str).tolist()
    color_data = color_df["revenue"].tolist()
    
    print("Color share calculated.")

    data = {
        "kpis": {
            "total_sales": round(total_sales, 2),
            "total_quantity": int(total_qty),
            "total_orders": int(total_orders),
            "aov": round(aov, 2)
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
    
    print("Data dictionary formatting done.")
    
    # 5. Generate Insights
    # Mocking insights_helper locally to test separation
    def generate_insights(data):
        insights = []
        # 1. KPI Insight
        total_sales = data['kpis']['total_sales']
        total_orders = data['kpis']['total_orders']
        if total_sales > 0:
            insights.append(f"Total revenue generated is **₹{total_sales:,.2f}** from **{total_orders:,}** orders.")

        # 2. Trend Insight (Peak Month)
        trend_sales = data['trend']['sales']
        trend_labels = data['trend']['labels']
        if trend_sales:
            max_sales = max(trend_sales)
            max_idx = trend_sales.index(max_sales)
            peak_month = trend_labels[max_idx]
            insights.append(f"The highest performing month was **{peak_month}** with **₹{max_sales:,.2f}** in sales.")

        # 3. Size Insight
        size_labels = data['size_share']['labels']
        size_data = data['size_share']['data']
        if size_data:
            top_size_idx = size_data.index(max(size_data))
            top_size = size_labels[top_size_idx]
            top_size_val = size_data[top_size_idx]
            insights.append(f"**{top_size}** is the most popular size, contributing **₹{top_size_val:,.2f}** to total revenue.")

        # 4. Color Insight
        color_labels = data['color_share']['labels']
        color_data = data['color_share']['data']
        if color_data:
            top_color_idx = color_data.index(max(color_data))
            top_color = color_labels[top_color_idx]
            top_color_val = color_data[top_color_idx]
            insights.append(f"**{top_color}** is the top-selling color variant, generating **₹{top_color_val:,.2f}**.")
        
        return insights

    print("Generating insights...")
    insights = generate_insights(data)
    # Remove special chars for console printing
    safe_insights = [s.replace('\u20b9', 'INR ') for s in insights]
    print("Insights generated:", safe_insights)
    
    import json
    # Custom encoder to handle NaN
    print("Attempting JSON dump...")
    json_str = json.dumps(data) 
    print("JSON dump successful.")
    
    # Check for NaNs explicitly
    import math
    def check_nan(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                check_nan(v, path + "." + k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                check_nan(v, f"{path}[{i}]")
        elif isinstance(obj, float):
            if math.isnan(obj):
                print(f"FOUND NAN at {path}")
            if math.isinf(obj):
                print(f"FOUND INF at {path}")
    
    print("Checking for NaNs...")
    check_nan(data)
    print("NaN check complete.")

    print("SUCCESS")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

