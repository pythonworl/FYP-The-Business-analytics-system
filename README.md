# Business Analytics Predictor

This is a web application built with **FastAPI** to predict:
1.  **Demand**: Monthly sales quantity based on historical trends.
2.  **Sales Revenue**: Per-order revenue based on price, discount, and location.

## How to Run Locally

1.  **Install Dependencies**:
    ```bash
    pip install fastapi uvicorn pandas joblib jinja2 python-multipart
    ```

2.  **Run the Server**:
    ```bash
    uvicorn main:app --reload
    ```
    Access the app at `http://127.0.0.1:8000`.

## GitHub Repository
https://github.com/pythonworl/FYP-The-Business-analytics-system
