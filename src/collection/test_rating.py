import requests


BASE_URL = "https://codeforces.com/api"


def get_rating_history(handle: str):
    response = requests.get(
        f"{BASE_URL}/user.rating",
        params={"handle": handle},
        timeout=30,
    )

    print("HTTP status:", response.status_code)
    response.raise_for_status()

    data = response.json()

    print("status:", data["status"])
    print("number of contests:", len(data["result"]))

    for contest in data["result"][:5]:
        print(contest)


if __name__ == "__main__":
    get_rating_history("tourist")