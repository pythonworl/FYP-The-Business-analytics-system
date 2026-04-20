def generate_insights(data, source="sales"):
    """
    Generates dynamic business insights based on the report data structure.
    data: { "kpis": {...}, "charts": {...} }
    source: 'sales', 'churn', 'report'
    """
    insights = []
    kpis = data.get("kpis", {})
    charts = data.get("charts", {})
    
    if source == "sales":
        # 1. Total Revenue Check
        if "Total Revenue" in kpis:
            insights.append(f"Sales performance check: **Total Revenue** is currently **{kpis['Total Revenue']}**.")
        
        # 2. Category Share Calculation
        donut = charts.get("donut", {})
        if donut and "data" in donut:
            vals = donut["data"]
            labels = donut["labels"]
            if vals:
                total_v = sum(vals)
                top_v = max(vals)
                pct = (top_v / total_v) * 100 if total_v > 0 else 0
                leader = labels[vals.index(top_v)]
                insights.append(f"Category Dominance: **{leader}** accounts for **{pct:.1f}%** of your total revenue segment.")
        
        # 3. Peak Seasonality
        trend = charts.get("trend", {})
        if trend and "data" in trend:
            vals = trend["data"]
            labels = trend["labels"]
            if vals and len(vals) > 1:
                peak_idx = vals.index(max(vals))
                peak_v = vals[peak_idx]
                peak_label = labels[peak_idx]
                insights.append(f"Peak Seasonality identified: **{peak_label}** was your strongest month with volume of **{peak_v:,.2f}**.")
        
        # 4. Regional Opportunity
        bar = charts.get("bar", {})
        if bar and "data" in bar:
            vals = bar["data"]
            labels = bar["labels"]
            if vals:
                best_idx = vals.index(max(vals))
                region = labels[best_idx]
                insights.append(f"Region Leader: **{region}** shows the strongest market penetration in this dataset.")

    elif source == "churn":
        # 1. Churn Rate Alert
        if "Predicted Churn Rate" in kpis:
            insights.append(f"Risk Analysis: Current **Churn Rate** is projected at **{kpis['Predicted Churn Rate']}** across your userbase.")
            
        # 2. Satisfaction Correlation
        if "Avg Satisfaction" in kpis:
            sat = float(kpis["Avg Satisfaction"].split(" / ")[0])
            if sat < 3.5:
                insights.append(f"Critical Warning: **Customer Satisfaction ({sat}/5)** is below target threshold; prioritze quality feedback immediately.")
            else:
                 insights.append(f"Satisfaction Health: Average score of **{sat}/5** indicates a positive sentiment trend in returning users.")
        
        # 3. Loyalty Impact
        donut = charts.get("donut", {})
        if donut and "data" in donut:
            vals = donut["data"]
            labels = donut["labels"]
            if len(vals) >= 2:
                enrollment_pct = (vals[1] / sum(vals)) * 100 if sum(vals) > 0 else 0
                insights.append(f"Loyalty Check: **{enrollment_pct:.1f}%** of customers are enrolled in your program; targeted conversion is recommended.")

    if not insights:
        insights = ["Upload more business data to unlock deeper AI-driven strategic insights."]
        
    return insights
