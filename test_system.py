import requests

def test_api():
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            print("API: Działa")
        else:
            print("API: Problem")
    except:
        print("API: Niedostępne")

if __name__ == "__main__":
    test_api()