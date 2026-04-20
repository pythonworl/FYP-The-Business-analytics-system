import pandas as pd
import numpy as np
import re
import google.generativeai as genai
import json
import os
import logging

class ChatAssistant:
    def __init__(self, df, sales_model, qty_model, churn_model):
        self.df = df
        self.sales_model = sales_model
        self.qty_model = qty_model
        self.churn_model = churn_model
        
        # Initialize Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.model = None
            logging.warning("GEMINI_API_KEY not found in environment.")

        # Smart Column Detection
        self.cols = self._detect_columns()
        
        # Precompute Category -> Sub-Category Mapping
        self.cat_to_sub = {}
        if self.cols['cat'] in df.columns and self.cols['sub'] in df.columns:
            for cat in df[self.cols['cat']].dropna().unique():
                subs = df[df[self.cols['cat']] == cat][self.cols['sub']].dropna().unique().tolist()
                self.cat_to_sub[str(cat)] = [str(s) for s in subs]
        
        self.categories = list(self.cat_to_sub.keys())
        self.subcategories = df[self.cols['sub']].dropna().unique().tolist() if self.cols['sub'] in df.columns else []
        self._prepare_recommendations()

    def _detect_columns(self):
        cols = {
            'sales': 'Sales', 'cat': 'Category', 'sub': 'Sub-Category', 'reg': 'Region', 'date': 'Order Date'
        }
        low_cols = {str(c).lower().strip(): c for c in self.df.columns}
        for key in ['sales', 'revenue', 'total_sales']:
            if key in low_cols: cols['sales'] = low_cols[key]; break
        for key in ['region', 'location', 'country', 'zone']:
            if key in low_cols: cols['reg'] = low_cols[key]; break
        for key in ['order date', 'date', 'datetime']:
            if key in low_cols: cols['date'] = low_cols[key]; break
        for key in ['category', 'product_category']:
            if key in low_cols: cols['cat'] = low_cols[key]; break
        for key in ['sub-category', 'subcategory', 'product_type']:
            if key in low_cols: cols['sub'] = low_cols[key]; break
        return cols

    def _prepare_recommendations(self):
        self.recommendations = {}
        sub_col = self.cols['sub']
        if sub_col not in self.df.columns: return
        order_col = next((c for c in self.df.columns if "Order ID" in c or "Invoice" in c), None)
        if order_col:
            orders = self.df.groupby(order_col)[sub_col].apply(list)
            co_occur = {}
            for items in orders:
                unique_items = list(set(items))
                for i in range(len(unique_items)):
                    for j in range(i + 1, len(unique_items)):
                        pair = tuple(sorted([str(unique_items[i]), str(unique_items[j])]))
                        co_occur[pair] = co_occur.get(pair, 0) + 1
            for (a, b), count in co_occur.items():
                if a not in self.recommendations: self.recommendations[a] = []
                if b not in self.recommendations: self.recommendations[b] = []
                self.recommendations[a].append((b, count)); self.recommendations[b].append((a, count))
            for item in self.recommendations:
                self.recommendations[item].sort(key=lambda x: x[1], reverse=True)
                self.recommendations[item] = [x[0] for x in self.recommendations[item][:3]]

    def handle_message(self, message: str):
        s_col = self.cols['sales']
        r_col = self.cols['reg']
        c_col = self.cols['cat']
        sb_col = self.cols['sub']

        if s_col not in self.df.columns:
            return {"reply": "I couldn't identify a sales/revenue column in your data to analyze."}

        # --- DATA CROSS-SECTIONAL ANALYSIS (PRECISION FIX) ---
        total_rev = self.df[s_col].sum()
        
        # 1. Revenue by Region/Location
        reg_perf = self.df.groupby(r_col)[s_col].sum().sort_values(ascending=False) if r_col in self.df.columns else {}
        reg_str = "\n".join([f"- {r}: {v:,.2f}" for r, v in reg_perf.items()])

        # 2. Revenue by Category
        cat_perf = self.df.groupby(c_col)[s_col].sum().sort_values(ascending=False)
        cat_str = "\n".join([f"- {c}: {v:,.2f}" for c, v in cat_perf.items()])

        # 3. CROSS-SECTIONAL DATA (Specific Category/Sub-Category in Specific Location)
        cross_str = "No granular cross-sectional data available."
        if r_col in self.df.columns:
            # Flexible grouping: Use Sub-Category if available, otherwise just Category
            group_cols = [r_col]
            if sb_col in self.df.columns:
                group_cols.append(sb_col)
            elif c_col in self.df.columns:
                group_cols.append(c_col)
            
            if len(group_cols) > 1:
                pivot = self.df.groupby(group_cols)[s_col].sum().reset_index()
                pivot = pivot.sort_values(by=s_col, ascending=False).head(50)
                # Build strings based on whether we have 2 columns
                lines = []
                for _, row in pivot.iterrows():
                    val_str = f"{row[group_cols[0]]} - {row[group_cols[1]]}: {row[s_col]:,.2f}"
                    lines.append(f"- {val_str}")
                cross_str = "\n".join(lines)

        ground_truth_context = (
            f"REAL BUSINESS DATA SUMMARY:\n"
            f"OVERALL REVENUE: {total_rev:,.2f}\n\n"
            f"REVENUE BY LOCATION:\n{reg_str}\n\n"
            f"REVENUE BY CATEGORY:\n{cat_str}\n\n"
            f"PRECISION BREAKDOWN (LOCATION - SEGMENT):\n{cross_str}\n"
        )

        low_msg = message.lower()
        intent = "GENERAL"
        if any(w in low_msg for w in ["what if", "simulate"]): intent = "WHAT_IF"
        elif any(w in low_msg for w in ["recommend", "suggest", "companion"]): intent = "RECOMMENDATION"

        # Check if user explicitly asked for advice/tips
        wants_tips = any(w in low_msg for w in ["tips", "advice", "suggestion", "how to improve", "strategy"])

        if intent == "WHAT_IF": return self._handle_what_if(message)
        elif intent == "RECOMMENDATION": return self._handle_recommendations(message)

        # --- Gemini Logic ---
        system_prompt = (
            "You are a Precision BI Assistant. Answer user questions based ONLY on the DATA SUMMARY provided.\n\n"
            "RULES:\n"
            "1. If the user asks for a specific figure (e.g., 'Amino Acid in UK'), look at the 'PRECISION BREAKDOWN' list first. Be exact.\n"
            "2. DO NOT provide professional advice, strategic tips, or detailed paragraphs unless the user EXPLICITLY asks for 'tips', 'advice', or 'suggestions'.\n"
            "3. If only a data question is asked, answer concisely with the number and a 1-sentence fact.\n"
            "4. If the user asks for tips/advice, provide DETAILED PROFESSIONAL PARAGRAPHS using your business intelligence.\n"
            "5. If a figure is not in the top summarized list, logically state: 'I can see the totals for this region, but a granular breakdown for that specific sub-category isn't in my immediate summary. However, [provide a related data point from the summary]'."
        )
        
        models_to_try = [
            'gemma-3-12b-it',
            'gemma-3-4b-it',
            'gemini-1.5-flash', 
            'gemini-2.0-flash',
            'gemini-2.5-flash'
        ]
        for m_name in models_to_try:
            try:
                model = genai.GenerativeModel(m_name)
                full_prompt = f"{system_prompt}\n\n{ground_truth_context}\n\nUser Question: {message}\n\nUser Wants Advice: {'Yes' if wants_tips else 'No'}"
                response = model.generate_content(full_prompt)
                return {"reply": response.text}
            except Exception as e:
                logging.error(f"Gemini attempt {m_name} failed: {e}")
                continue

        return {"reply": f"The top category is {cat_perf.idxmax()} with revenue of {cat_perf.max():,.2f}."}

    def _handle_what_if(self, msg):
        match = re.search(r"(\d+)%", msg)
        new_val = float(match.group(1)) / 100 if match else 0.20
        cat_col = self.cols['cat']
        s_col = self.cols['sales']
        category = next((str(c) for c in self.categories if str(c).lower() in msg.lower()), "Technology")
        
        baseline_X = pd.DataFrame([{
            "Category": category, "Sub-Category": self.cat_to_sub.get(category, ["General"])[0],
            "Region": "Central", "City": "Chicago", "Unit Price": 500.0, "Discount": 0.1,
            "Order_Year": 2024, "Order_Month": 12, "Order_Quarter": 4, "Quantity": 2,
        }])
        
        sim_X = baseline_X.copy()
        target = "discount"
        if "price" in msg.lower():
            sim_X["Unit Price"] *= (1.2 if "increase" in msg.lower() else 0.8)
            target = "price"
        else: sim_X["Discount"] = new_val

        try:
            base_pred = float(self.sales_model.predict(baseline_X)[0])
            sim_pred = float(self.sales_model.predict(sim_X)[0])
            diff = sim_pred - base_pred
            result = (f"Simulation for {category}: {target} change suggests a shift from {base_pred:,.2f} to {sim_pred:,.2f} ({diff:,.2f} impact).")
            model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = f"Explain this simulation concisely: {result}."
            return {"reply": model.generate_content(prompt).text}
        except: return {"reply": f"Simulation suggests revenue would move to {sim_pred:,.2f}."}

    def _handle_recommendations(self, msg):
        found_sub = next((str(sub) for sub in self.subcategories if str(sub).lower() in msg.lower()), None)
        if found_sub and found_sub in self.recommendations:
            recs = self.recommendations[found_sub]
            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
                prompt = f"Suggest these products in a detailed paragraph for {found_sub}: {', '.join(recs)}."
                return {"reply": model.generate_content(prompt).text}
            except: return {"reply": f"For {found_sub}, customers often buy: {', '.join(recs)}."}
        return {"reply": "I can suggest products. Which sub-category are you interested in?"}
