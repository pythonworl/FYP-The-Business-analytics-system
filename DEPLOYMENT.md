# Deployment Guide: Business Analytics System

This guide explains how to deploy your FastAPI application to **Vercel** or **Hugging Face Spaces**.

## Prerequisites
1. A **GitHub** account.
2. Your project pushed to a GitHub repository.

---

## Option 1: Deploying to Hugging Face Spaces (Recommended)
Hugging Face Spaces provides 16GB of RAM for free, which is ideal for your Machine Learning models.

### Steps:
1. Go to [huggingface.co](https://huggingface.co/) and log in.
2. Click **New Space**.
3. **Name**: `business-analytics-system` (or anything you like).
4. **SDK**: Select **Docker**.
5. **Template**: Choose **Blank**.
6. **Visibility**: Public (Free).
7. Click **Create Space**.
8. **Settings**: Go to the "Settings" tab of your new Space.
9. **Variables and Secrets**:
   - Add a **New Secret**:
     - **Name**: `GEMINI_API_KEY`
     - **Value**: `[Your API Key]`
10. **Files and Versions**:
    - Upload all your project files (including the `Dockerfile`, `requirements.txt`, and `.pkl` model files) to the repository.
    - Alternatively, connect your GitHub repo to the Space.
11. The Space will automatically build and start. Access it via the "App" tab.

---

## Option 2: Deploying to Vercel
Vercel is great for fast, serverless web apps.

### Steps:
1. Go to [vercel.com](https://vercel.com/) and log in.
2. Click **Add New** > **Project**.
3. Import your GitHub repository.
4. **Environment Variables**:
   - Add `GEMINI_API_KEY` with your key value.
5. Click **Deploy**.
6. **Note**: If the deployment fails due to "Bundle Size," you may need to use Option 1 (Hugging Face) because of the 215MB model file.

---

## Updating and Redeploying
Whenever you make changes to your code:
1. **Commit** your changes and **Push** them to GitHub.
2. Both platform will detect the push and **automatically redeploy** your update.

## Important Note on Data
Since these are free tiers, any data you upload or models you train **live** via the dashboard will be cleared when the app restarts. The base models and data you include in your GitHub repository will always be there.
