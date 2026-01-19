import requests
import time
import sys

BASE_URL = "http://127.0.0.1:8000"
CUSTOMER_ID = "C00001"

def wait_for_server():
    print("Waiting for server...")
    for _ in range(30):
        try:
            r = requests.get(f"{BASE_URL}/docs")
            if r.status_code == 200:
                print("Server is up!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    print("Server failed to start.")
    return False

def test_score():
    print(f"\nTesting /score_customer for {CUSTOMER_ID}...")
    try:
        r = requests.post(f"{BASE_URL}/score_customer", json={"customer_id": CUSTOMER_ID})
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print("Response:", r.json())
        else:
            print("Error:", r.text)
    except Exception as e:
        print(f"Exception: {e}")

def test_churn():
    print(f"\nTesting /predict_churn for {CUSTOMER_ID}...")
    try:
        r = requests.post(f"{BASE_URL}/predict_churn", json={"customer_id": CUSTOMER_ID, "horizon_days": 60})
        print(f"Status: {r.status_code}")
        print("Response:", r.json())
    except Exception as e:
        print(f"Exception: {e}")

def test_survival():
    print(f"\nTesting /predict_survival for {CUSTOMER_ID}...")
    try:
        r = requests.post(f"{BASE_URL}/predict_survival", json={"customer_id": CUSTOMER_ID})
        print(f"Status: {r.status_code}")
        # summary of response
        if r.status_code == 200:
            data = r.json()
            print(f"Expected Lifetime: {data.get('expected_remaining_lifetime')}")
            print(f"Curve Points: {len(data.get('survival_curve', []))}")
        else:
            print(r.text)
    except Exception as e:
        print(f"Exception: {e}")

def test_clv():
    print(f"\nTesting /estimate_clv (bgnbd) for {CUSTOMER_ID}...")
    try:
        r = requests.post(f"{BASE_URL}/estimate_clv", json={"customer_id": CUSTOMER_ID, "method": "bgnbd"})
        print(f"Status: {r.status_code}")
        print("Response:", r.json())
    except Exception as e:
        print(f"Exception: {e}")

    print(f"\nTesting /estimate_clv (survival) for {CUSTOMER_ID}...")
    try:
        r = requests.post(f"{BASE_URL}/estimate_clv", json={"customer_id": CUSTOMER_ID, "method": "survival"})
        print(f"Status: {r.status_code}")
        print("Response:", r.json())
    except Exception as e:
        print(f"Exception: {e}")

def test_rank():
    print(f"\nTesting /rank_customers_for_retention...")
    try:
        r = requests.post(f"{BASE_URL}/rank_customers_for_retention", json={"top_k": 5, "strategy": "high_clv_high_churn"})
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"Returned {len(data.get('customers', []))} customers")
            if data.get('customers'):
                print("First customer:", data['customers'][0])
        else:
            print(r.text)
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    if wait_for_server():
        test_score()
        test_churn()
        test_survival()
        test_clv()
        test_rank()
    else:
        sys.exit(1)
