import requests


BASE_URL = "https://codeforces.com/api"


def get_user_info(handle: str):
    response = requests.get(
        f"{BASE_URL}/user.info",
        params={"handles": handle},
        timeout=30,
    )

    print("HTTP status:", response.status_code)
    response.raise_for_status()

    data = response.json()

    print("status:", data["status"])
    print("result:")
    print(data["result"])


if __name__ == "__main__":
    get_user_info("tourist")