import os
import requests

BASE44_API_KEY = os.environ.get("BASE44_API_KEY")

def test_base44():
    url = "https://app.base44.com/api/apps/6981c49a70b17c150ed2d05b/entities/PlaybookRule"

    headers = {
        "api_key": BASE44_API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)

    return {
        "status_code": response.status_code,
        "data": response.json()
    }
