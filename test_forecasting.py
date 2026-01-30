import pandas as pd
from forecasting import forecast_sales

# Load data
df = pd.read_csv("Ecommerce_Sales_Data_Expanded.csv")

# Run forecast for "Furniture"
print("Testing Furniture Forecast (12 months)...")
result = forecast_sales(df, horizon=12, category="Furniture")

if "error" in result:
    print(f"Error: {result['error']}")
else:
    print(f"Success! Best Model: {result['model_name']}")
    print(f"MAPE: {result['mape']}%")
    print(f"Forecast Steps: {len(result['forecast']['values'])}")
    print(f"First 3 Forecast Values: {result['forecast']['values'][:3]}")

# Run forecast for ALL
print("\nTesting Global Forecast (6 months)...")
result_all = forecast_sales(df, horizon=6, category="All")
if "error" in result_all:
    print(f"Error: {result_all['error']}")
else:
    print(f"Success! Best Model: {result_all['model_name']}")
    print(f"MAPE: {result_all['mape']}%")
