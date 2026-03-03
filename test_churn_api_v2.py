import requests

url = "http://127.0.0.1:8000/api/predict/churn"
# New payload matching refined features
payload = {
    "Age": 30,
    "AnnualIncome": 50000,
    "NumberOfPurchases": 10,
    "TimeSpentOnWebsite": 15.5,
    "CustomerTenureYears": 2.5,
    "LastPurchaseDaysAgo": 30,
    "SessionCount": 5,
    "CustomerSatisfaction": 3,
    "DiscountsAvailed": 2,
    "LoyaltyProgram": 1
}

try:
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    print("Response:", response.text)
except Exception as e:
    print("Error:", e)
