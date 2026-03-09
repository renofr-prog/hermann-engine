import os
import requests

BASE44_API_KEY = os.getenv("BASE44_SERVICE_ROLE_KEY")
BASE44_URL = "https://base44.app/api"

def test_base44():
    try:
        headers = {
            "api_key": BASE44_API_KEY,
            "Content-Type": "application/json"
        }

        r = requests.get(
            f"{BASE44_URL}/health",
            headers=headers,
            timeout=10
        )

        return {
            "status_code": r.status_code,
            "response": r.text
        }

    except Exception as e:
        return {"error": str(e)}
