# 🚀 Business Analytics Predictor: Enterprise Edition

State-of-the-art Business Intelligence and Predictive Analytics platform powered by **FastAPI** and **Google Gemini AI**. This system transforms raw business data into actionable strategic insights through advanced Machine Learning and NL-powered intelligence.

---

## 🌟 Key Features

### 🧠 AI Intelligence Assistant (Gemini & Gemma 3)
- **Data-Aware Chat**: Ask natural questions about your specific datasets. The AI understands your columns (Location, Revenue, Category) regardless of naming.
- **Precision Data Awareness**: Leverages cross-sectional analysis (Category x Location) to provide exact figures for granular business segments.
- **Hybrid Intelligence**: Seamlessly fails over from Gemini 2.5/2.0 to **Gemma 3** to ensure 100% uptime regardless of API quota limits.
- **Strategic Consulting**: Generates detailed professional strategy paragraphs when asked for business advice or growth tips.

### 📈 Predictive Analytics Modules
- **Demand Forecast**: Predict monthly sales quantity across regions and categories using ARIMA/Random Forest.
- **Revenue Predictor**: Estimate per-order revenue based on real-time feature importance (Price, Discount, etc.).
- **Trend Horizon**: Project long-term micro sales movements month-over-month.
- **Churn Analytics**: Identify at-risk customers instantly through predictive demographic and behavioral metrics.

### 📊 Executive Board
- **Automated Insights**: Dynamic KPIs and beautiful visualization charts (Revenue Trends, Category Composition, Regional Distribution).
- **Automated AutoML Engine**: The backend automatically evaluates multiple algorithms and selects the top-performing model for your specific data.

---

## 🛠️ Technical Stack
- **Backend**: FastAPI (Python 3.10+)
- **AI Core**: Google Gemini SDK (Gemma 3, Gemini 1.5/2.0/2.5)
- **Data Engine**: Pandas, Scikit-learn, Joblib
- **Frontend**: Vanilla JS, Chart.js, Premium Dark UI

---

## 🚀 Getting Started

### 1. Requirements
Ensure you have Python 3.10+ installed.

### 2. Installation
```bash
pip install fastapi uvicorn pandas joblib jinja2 python-multipart google-generativeai python-dotenv scikit-learn
```

### 3. Environment Setup
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 4. Launch Application
```bash
uvicorn main:app --reload
```
Access the dashboard at `http://127.0.0.1:8000`.

---

## 📂 Project Structure
- `main.py`: Core FastAPI application and API routes.
- `chat_helper.py`: AI Intelligence Assistant logic (Gemini/Gemma integration).
- `trainer.py`: AutoML engine for model training and feature discovery.
- `insights_helper.py`: Automated business insight generation logic.
- `templates/`: Professional frontend layouts.
- `static/`: Modern styles and interactive dashboard JS.

---

## 🔗 Repository
[GitHub: pythonworl/FYP-The-Business-analytics-system](https://github.com/pythonworl/FYP-The-Business-analytics-system)
