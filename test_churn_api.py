import requests

url = "http://127.0.0.1:8000/api/predict/churn"
payload = {
    "Age": 30,
    "AnnualIncome": 50000,
    "NumberOfPurchases": 10,
    "TimeSpentOnWebsite": 15.5,
    "CustomerTenureYears": 2.5,
    "LastPurchaseDaysAgo": 30,
    "SessionCount": 5,
    "Gender": "Male",
    "ProductCategory": "Electronics",
    "PreferredDevice": "Desktop",
    "Region": "North",
    "ReferralSource": "Organic",
    "CustomerSegment": "Regular"
}

try:
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    print("Response:", response.text)
except Exception as e:
    print("Error:", e)
