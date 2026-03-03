
def generate_insights(data):
    """
    Generates a list of verbal insights/strings based on the report data.
    data structure: {
        "kpis": {...},
        "trend": {...},
        "size_share": {...},
        "color_share": {...}
    }
    """
    insights = []

    # 1. KPI Insight
    total_sales = data['kpis']['total_sales']
    total_orders = data['kpis']['total_orders']
    if total_sales > 0:
        insights.append(f"Total revenue generated is **{total_sales:,.2f}** from **{total_orders:,}** orders.")

    # 2. Trend Insight (Peak Month)
    trend_sales = data['trend']['sales']
    trend_labels = data['trend']['labels']
    if trend_sales:
        max_sales = max(trend_sales)
        max_idx = trend_sales.index(max_sales)
        peak_month = trend_labels[max_idx]
        insights.append(f"The highest performing month was **{peak_month}** with **{max_sales:,.2f}** in sales.")

    # 3. Size Insight
    size_labels = data['size_share']['labels']
    size_data = data['size_share']['data']
    if size_data:
        top_size_idx = size_data.index(max(size_data))
        top_size = size_labels[top_size_idx]
        top_size_val = size_data[top_size_idx]
        insights.append(f"**{top_size}** is the most popular size, contributing **{top_size_val:,.2f}** to total revenue.")

    # 4. Color Insight
    color_labels = data['color_share']['labels']
    color_data = data['color_share']['data']
    if color_data:
        top_color_idx = color_data.index(max(color_data))
        top_color = color_labels[top_color_idx]
        top_color_val = color_data[top_color_idx]
        insights.append(f"**{top_color}** is the top-selling color variant, generating **{top_color_val:,.2f}**.")
    
    return insights
