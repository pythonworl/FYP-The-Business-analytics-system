import requests

try:
    response = requests.get('http://127.0.0.1:8000/')
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        html = response.text
        if 'id="btnSales"' in html:
            print("Found btnSales in HTML")
        else:
            print("btnSales NOT found in HTML")
        
        if '<script src="/static/app.js"></script>' in html:
             print("Found app.js script tag")
        else:
             print("app.js script tag NOT found")

except Exception as e:
    print(f"Error: {e}")
