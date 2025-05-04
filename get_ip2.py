import requests

def get_public_ip():
    try:
        response = requests.get("https://api.ipify.org")
        return response.text
    except:
        try:
            response = requests.get("https://httpbin.org/ip")
            return response.json()["origin"]
        except:
            return "无法获取公网 IP"

print("公网 IP:", get_public_ip())