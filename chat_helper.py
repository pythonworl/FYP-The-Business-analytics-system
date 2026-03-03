import pandas as pd
import numpy as np
import re
import ollama
import json

class ChatAssistant:
    def __init__(self, df, sales_model, qty_model, churn_model):
        self.df = df
        self.sales_model = sales_model
        self.qty_model = qty_model
        self.churn_model = churn_model
        # Precompute mapping: Category -> list of Sub-Categories
        self.cat_to_sub = {}
        for cat in df["Category"].dropna().unique():
            subs = df[df["Category"] == cat]["Sub-Category"].dropna().unique().tolist()
            self.cat_to_sub[cat] = subs
        
        self.categories = list(self.cat_to_sub.keys())
        self.subcategories = df["Sub-Category"].dropna().unique().tolist()
        self._prepare_recommendations()

    def _prepare_recommendations(self):
        """Precompute a simple sub-category co-occurrence matrix."""
        self.recommendations = {}
        order_col = "Order ID" if "Order ID" in self.df.columns else "Invoice ID" if "Invoice ID" in self.df.columns else None
        
        if order_col:
            orders = self.df.groupby(order_col)["Sub-Category"].apply(list)
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
                self.recommendations[a].append((b, count))
                self.recommendations[b].append((a, count))
            
            for item in self.recommendations:
                self.recommendations[item].sort(key=lambda x: x[1], reverse=True)
                self.recommendations[item] = [x[0] for x in self.recommendations[item][:3]]

    def handle_message(self, message: str):
        """
        Agentic flow: 
        1. Get Ground-Truth from Python.
        2. Categorize intent.
        3. Use LLM to format response based ONLY on Ground-Truth.
        """
        # --- Pre-calculate Ground Truth (The facts) ---
        top_cat = self.df.groupby("Category")["Sales"].sum().idxmax()
        top_cat_val = self.df.groupby("Category")["Sales"].sum().max()
        top_sub = self.df.groupby("Sub-Category")["Sales"].sum().idxmax()
        top_sub_val = self.df.groupby("Sub-Category")["Sales"].sum().max()
        top_reg = self.df.groupby("Region")["Sales"].sum().idxmax()
        
        # Build a clear map for the AI
        mapping_str = "\n".join([f"- {c}: {', '.join(s)}" for c, s in self.cat_to_sub.items()])

        # New: Get top sub-categories in each category for deeper intelligence
        sub_perf = self.df.groupby(["Category", "Sub-Category"])["Sales"].sum().sort_values(ascending=False)
        sub_perf_str = "\n".join([f"- {cat} > {sub}: {val:,.2f}" for (cat, sub), val in sub_perf.head(10).items()])

        ground_truth_context = (
            f"REAL BUSINESS DATA (USE ONLY THESE FIGURES):\n"
            f"- Overall Top Category: {top_cat} (Sales: {top_cat_val:,.2f})\n"
            f"- Overall Top Sub-Category: {top_sub} (Sales: {top_sub_val:,.2f})\n"
            f"- Leading Region: {top_reg}\n"
            f"TOP 10 SUB-CATEGORY PERFORMANCE:\n{sub_perf_str}\n"
            f"CATEGORY TO PRODUCT MAPPING:\n{mapping_str}\n"
        )

        low_msg = message.lower()
        
        # Determine intent (Simplified for speed)
        intent = "GENERAL"
        if "what if" in low_msg or "simulate" in low_msg: intent = "WHAT_IF"
        elif "recommend" in low_msg or "suggest" in low_msg: intent = "RECOMMENDATION"
        elif any(word in low_msg for word in ["category", "subcategory", "region", "top", "best", "sales"]): intent = "ANALYTICS"

        # Handle via specific logic but keep LLM in the loop for 'flow'
        if intent == "WHAT_IF":
            return self._handle_what_if(message)
        elif intent == "RECOMMENDATION":
            return self._handle_recommendations(message)
        
        # --- Analytics & General ---
        # We give the LLM the context so it CANNOT hallucinate names like 'Electronics' 
        # unless they are actually in the context.
        system_prompt = (
            "You are a helpful and professional Business Intelligence Assistant. Your goal is to explain the provided REAL BUSINESS DATA to the user. "
            "1. Use the REAL BUSINESS DATA to answer the question accurately.\n"
            "2. Be conversational but concise. Don't just give one-word answers; provide a professional summary.\n"
            "3. CRITICAL: Never invent categories, percentages, or numbers NOT found in the REAL BUSINESS DATA.\n"
        )
        
        try:
            response = ollama.chat(model='llama3', messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'system', 'content': ground_truth_context},
                {'role': 'user', 'content': message},
            ])
            reply = response['message']['content']
            
            # Final Safety Catch: If the LLM still somehow hallucinates a wrong top category
            if intent == "ANALYTICS" and "category" in low_msg and str(top_cat).lower() not in reply.lower():
                return {"reply": f"Based on the data, your top-performing category is {top_cat} with total sales of {top_cat_val:,.2f}."}
            
            return {"reply": reply}
        except Exception as e:
            return {"reply": f"According to your records, the top category is {top_cat} ({top_cat_val:,.2f})."}

    def _handle_what_if(self, msg):
        # Deterministic simulation logic
        match = re.search(r"(\d+)%", msg)
        new_val = float(match.group(1)) / 100 if match else 0.20
        category = "Technology" if "tech" in msg.lower() else "Furniture" if "furn" in msg.lower() else "Office Supplies"
        
        baseline_X = pd.DataFrame([{
            "Category": category, "Sub-Category": "Accessories" if category == "Technology" else "Chairs",
            "Region": "Central", "City": "Chicago", "Unit Price": 500.0, "Discount": 0.1,
            "Order_Year": 2024, "Order_Month": 12, "Order_Quarter": 4, "Quantity": 2,
        }])
        
        sim_X = baseline_X.copy()
        target = "discount"
        if "price" in msg.lower():
            sim_X["Unit Price"] *= (1.2 if "increase" in msg.lower() else 0.8)
            target = "price"
        else:
            sim_X["Discount"] = new_val

        base_pred = self.sales_model.predict(baseline_X)[0]
        sim_pred = self.sales_model.predict(sim_X)[0]
        diff = sim_pred - base_pred
        
        raw_result = (f"Simulated {target} change to {new_val*100:.0f}%. "
                      f"Original: {base_pred:,.2f}, New: {sim_pred:,.2f}, Impact: {diff:,.2f}")

        try:
            nice_reply = ollama.chat(model='llama3', messages=[
                {'role': 'system', 'content': "You are an analyst. Explain this simulation result professionally. Use ONLY the numbers provided."},
                {'role': 'user', 'content': raw_result},
            ])
            return {"reply": nice_reply['message']['content']}
        except:
            return {"reply": f"Simulation for {category}: New predicted sales {sim_pred:,.2f}."}

    def _handle_recommendations(self, msg):
        found_sub = None
        for sub in self.subcategories:
            if str(sub).lower() in msg.lower():
                found_sub = str(sub)
                break
        
        if found_sub and found_sub in self.recommendations:
            recs = self.recommendations[found_sub]
            try:
                nice_reply = ollama.chat(model='llama3', messages=[
                    {'role': 'system', 'content': f"Customers who bought {found_sub} also liked {', '.join(recs)}. Suggest these naturally."},
                    {'role': 'user', 'content': msg},
                ])
                return {"reply": nice_reply['message']['content']}
            except:
                return {"reply": f"Recommended for {found_sub}: {', '.join(recs)}."}
        
        return {"reply": "I can suggest companion products. Which sub-category are you interested in (e.g., Chairs, Phones)?"}
