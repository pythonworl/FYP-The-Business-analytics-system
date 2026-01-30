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

## How to Upload to GitHub

Follow these steps to push your code to GitHub:

1.  **Initialize Git**:
    ```bash
    git init
    ```

2.  **Add Files**:
    (Note: Large `.pkl` models and data files are excluded via `.gitignore` to prevent errors)
    ```bash
    git add .
    ```

3.  **Commit**:
    ```bash
    git commit -m "Initial commit"
    ```

4.  **Create Repository on GitHub**:
    - Go to https://github.com/new
    - Create a new repository (e.g., `business-analytics-predictor`)
    - Do **not** initialize with README/license.

5.  **Push Code**:
    Replace `YOUR_USERNAME` with your GitHub username:
    ```bash
    git branch -M main
    git remote add origin https://github.com/YOUR_USERNAME/business-analytics-predictor.git
    git push -u origin main
    ```
