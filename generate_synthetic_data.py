import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ----------------------------
# Config
# ----------------------------
INPUT_CSV = "Ecommerce_Sales_Data_2024_2025.csv"   # your current file
OUTPUT_CSV = "Ecommerce_Sales_Data_Expanded.csv"   # new expanded file

TARGET_ROWS = 80000   # try 50k, 80k, 100k
START_YEAR = 2022     # extend history backward
END_YEAR = 2025       # extend forward (inclusive)
RANDOM_SEED = 42

# If you want more/less seasonality effect:
SEASONAL_STRENGTH = 0.30  # 0.0=no seasonality, 0.2-0.4 moderate

np.random.seed(RANDOM_SEED)


# ----------------------------
# Helpers
# ----------------------------
def safe_to_datetime(s):
    return pd.to_datetime(s, errors="coerce")

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def weighted_choice(values, probs, size=1):
    probs = np.array(probs, dtype=float)
    probs = probs / probs.sum()
    return np.random.choice(values, size=size, p=probs)

def month_weights_from_data(df):
    # Base weights from historical frequency
    month_counts = df["Order_Month"].value_counts().sort_index()
    months = np.arange(1, 13)
    base = np.array([month_counts.get(m, 1) for m in months], dtype=float)

    # Add mild seasonality bump (e.g., higher around Nov/Dec, etc.)
    seasonal = np.ones(12, dtype=float)
    for m in [11, 12]:
        seasonal[m-1] += SEASONAL_STRENGTH
    for m in [2, 3]:
        seasonal[m-1] -= SEASONAL_STRENGTH / 2

    w = base * seasonal
    w = w / w.sum()
    return months, w


def build_segment_stats(df):
    """
    Build per-segment distributions for Unit Price, Discount, Quantity.
    Segment = (Category, Sub-Category, Region)
    """
    seg_cols = ["Category", "Sub-Category", "Region"]

    for c in seg_cols:
        df[c] = df[c].astype(str)

    seg = df.groupby(seg_cols).agg(
        price_med=("Unit Price", "median"),
        price_q1=("Unit Price", lambda x: np.percentile(x, 25)),
        price_q3=("Unit Price", lambda x: np.percentile(x, 75)),

        disc_med=("Discount", "median"),
        disc_q1=("Discount", lambda x: np.percentile(x, 25)),
        disc_q3=("Discount", lambda x: np.percentile(x, 75)),

        qty_med=("Quantity", "median"),
        qty_q1=("Quantity", lambda x: np.percentile(x, 25)),
        qty_q3=("Quantity", lambda x: np.percentile(x, 75)),

        sales_med=("Sales", "median") if "Sales" in df.columns else ("Unit Price", "median"),
    ).reset_index()

    def iqr_scale(q1, q3):
        s = (q3 - q1)
        return s if s > 1e-9 else max(q3 * 0.10, 1.0)

    seg["price_scale"] = seg.apply(lambda r: iqr_scale(r["price_q1"], r["price_q3"]), axis=1)
    seg["disc_scale"] = seg.apply(lambda r: iqr_scale(r["disc_q1"], r["disc_q3"]), axis=1)
    seg["qty_scale"]  = seg.apply(lambda r: iqr_scale(r["qty_q1"],  r["qty_q3"]), axis=1)

    return seg


def sample_normal_around(med, scale, minv=None, maxv=None):
    x = np.random.normal(loc=med, scale=scale)
    if minv is not None:
        x = max(minv, x)
    if maxv is not None:
        x = min(maxv, x)
    return x


def generate_order_id(start_id, n):
    return np.arange(start_id, start_id + n, dtype=int)


def make_customer_names(n):
    """
    FIXED: use np.char.add for safe vectorized string concat in NumPy.
    """
    first = np.array(["Amina", "Sara", "Ali", "Omar", "Zain", "Nadia", "Riya", "Ishan",
                      "Misha", "Anika", "Ayaan", "Kashvi"], dtype=str)
    last  = np.array(["Khan", "Sharma", "Patel", "Singh", "Desai", "Gupta", "Acharya",
                      "Chandra", "Thakur", "Ram", "Yadav"], dtype=str)

    fn = np.random.choice(first, size=n)
    ln = np.random.choice(last, size=n)

    full = np.char.add(np.char.add(fn, " "), ln)
    return full.astype(str)


# ----------------------------
# Main
# ----------------------------
def main():
    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find {INPUT_CSV} in this folder.")

    df = pd.read_csv(input_path)

    required = ["Order Date", "Customer Name", "Region", "City", "Category", "Sub-Category",
                "Product Name", "Quantity", "Unit Price", "Discount", "Sales", "Profit", "Payment Mode"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("Missing columns:", missing)
        print("Proceeding anyway, but output may be incomplete.")

    df["Order Date"] = safe_to_datetime(df["Order Date"])
    df = df.dropna(subset=["Order Date"]).copy()
    df["Order_Year"] = df["Order Date"].dt.year.astype(int)
    df["Order_Month"] = df["Order Date"].dt.month.astype(int)

    for col in ["Quantity", "Unit Price", "Discount", "Sales", "Profit"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[c for c in ["Quantity", "Unit Price", "Discount"] if c in df.columns]).copy()

    months, month_probs = month_weights_from_data(df)

    years = np.arange(START_YEAR, END_YEAR + 1)
    existing_years = df["Order_Year"].value_counts().to_dict()
    year_base = np.array([existing_years.get(y, 1) for y in years], dtype=float)
    year_probs = year_base / year_base.sum()

    payment_modes = df["Payment Mode"].dropna().astype(str).unique().tolist() if "Payment Mode" in df.columns else ["Card"]
    pm_counts = df["Payment Mode"].value_counts() if "Payment Mode" in df.columns else pd.Series([1], index=payment_modes)
    pm_probs = np.array([pm_counts.get(pm, 1) for pm in payment_modes], dtype=float)
    pm_probs = pm_probs / pm_probs.sum()

    city_by_region = {}
    if "City" in df.columns:
        for r, sub in df.groupby("Region"):
            counts = sub["City"].value_counts()
            cities = counts.index.astype(str).tolist()
            probs = (counts.values / counts.values.sum()).astype(float)
            city_by_region[str(r)] = (cities, probs)

    prod_by_subcat = {}
    if "Product Name" in df.columns:
        for (cat, subcat), sub in df.groupby(["Category", "Sub-Category"]):
            counts = sub["Product Name"].value_counts()
            prods = counts.index.astype(str).tolist()
            probs = (counts.values / counts.values.sum()).astype(float)
            prod_by_subcat[(str(cat), str(subcat))] = (prods, probs)

    seg_stats = build_segment_stats(df)

    seg_freq = df.groupby(["Category", "Sub-Category", "Region"]).size().reset_index(name="n")
    seg_stats = seg_stats.merge(seg_freq, on=["Category", "Sub-Category", "Region"], how="left")
    seg_stats["n"] = seg_stats["n"].fillna(1).astype(int)
    seg_probs = seg_stats["n"].values.astype(float)
    seg_probs = seg_probs / seg_probs.sum()

    start_id = int(df["Order ID"].max() + 1) if "Order ID" in df.columns else 100000

    n_new = max(0, TARGET_ROWS - len(df))
    print(f"Original rows: {len(df)}")
    print(f"Generating additional rows: {n_new}")
    if n_new == 0:
        print("TARGET_ROWS <= current rows. Nothing to generate.")
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved: {OUTPUT_CSV}")
        return

    seg_idx = np.random.choice(np.arange(len(seg_stats)), size=n_new, p=seg_probs)
    seg_chosen = seg_stats.iloc[seg_idx].reset_index(drop=True)

    years_sampled = weighted_choice(years, year_probs, size=n_new)
    months_sampled = weighted_choice(np.arange(1, 13), month_probs, size=n_new)

    days = np.random.randint(1, 29, size=n_new)
    order_dates = [datetime(int(y), int(m), int(d)).strftime("%Y-%m-%d")
                   for y, m, d in zip(years_sampled, months_sampled, days)]

    regions = seg_chosen["Region"].astype(str).values
    cities = []
    for r in regions:
        if r in city_by_region:
            c_list, c_probs = city_by_region[r]
            cities.append(weighted_choice(c_list, c_probs, size=1)[0])
        else:
            cities.append(str(df["City"].dropna().sample(1).iloc[0]) if "City" in df.columns else "Unknown")

    cats = seg_chosen["Category"].astype(str).values
    subcats = seg_chosen["Sub-Category"].astype(str).values
    products = []
    for cat, subcat in zip(cats, subcats):
        key = (cat, subcat)
        if key in prod_by_subcat:
            p_list, p_probs = prod_by_subcat[key]
            products.append(weighted_choice(p_list, p_probs, size=1)[0])
        else:
            products.append(str(df["Product Name"].dropna().sample(1).iloc[0]) if "Product Name" in df.columns else "Product")

    unit_prices = []
    discounts = []
    quantities = []
    for i, row in seg_chosen.iterrows():
        up = sample_normal_around(row["price_med"], row["price_scale"], minv=1.0)
        disc = sample_normal_around(row["disc_med"], row["disc_scale"], minv=0.0, maxv=60.0)
        q = sample_normal_around(row["qty_med"], row["qty_scale"], minv=1.0)

        m = int(months_sampled[i])
        if m in [11, 12]:
            q *= (1.0 + SEASONAL_STRENGTH / 2)
        if m in [2, 3]:
            q *= (1.0 - SEASONAL_STRENGTH / 3)

        unit_prices.append(float(up))
        discounts.append(float(disc))
        quantities.append(int(round(q)))

    unit_prices = np.array(unit_prices)
    discounts = np.array(discounts)
    quantities = np.array(quantities)

    sales = unit_prices * quantities * (1.0 - discounts / 100.0)

    if "Profit" in df.columns and "Sales" in df.columns:
        df_nonzero = df[(df["Sales"] > 0) & (df["Profit"].notna())].copy()
        df_nonzero["margin"] = df_nonzero["Profit"] / df_nonzero["Sales"]
        margin_by_cat = df_nonzero.groupby("Category")["margin"].median().to_dict()
    else:
        margin_by_cat = {}

    profits = []
    for cat, s in zip(cats, sales):
        base_margin = margin_by_cat.get(cat, 0.15)
        noise = np.random.normal(0, 0.04)
        m = clamp(base_margin + noise, 0.01, 0.40)
        profits.append(float(s * m))
    profits = np.array(profits)

    pm = weighted_choice(payment_modes, pm_probs, size=n_new)
    cust_names = make_customer_names(n_new)
    order_ids = generate_order_id(start_id, n_new)

    new_df = pd.DataFrame({
        "Order ID": order_ids,
        "Order Date": order_dates,
        "Customer Name": cust_names,
        "Region": regions,
        "City": cities,
        "Category": cats,
        "Sub-Category": subcats,
        "Product Name": products,
        "Quantity": quantities,
        "Unit Price": np.round(unit_prices, 2),
        "Discount": np.round(discounts, 2),
        "Sales": np.round(sales, 2),
        "Profit": np.round(profits, 2),
        "Payment Mode": pm
    })

    out = pd.concat([df.drop(columns=["Order_Year", "Order_Month"], errors="ignore"), new_df], ignore_index=True)

    out["Order Date"] = safe_to_datetime(out["Order Date"])
    out = out.sort_values("Order Date").reset_index(drop=True)
    out["Order Date"] = out["Order Date"].dt.strftime("%Y-%m-%d")

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved expanded dataset: {OUTPUT_CSV}")
    print(f"Total rows now: {len(out)}")


if __name__ == "__main__":
    main()
