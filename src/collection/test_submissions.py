import requests


BASE_URL = "https://codeforces.com/api"


def get_submissions(handle: str):
    response = requests.get(
        f"{BASE_URL}/user.status",
        params={
            "handle": handle,
        },
        timeout=30,
    )

    print("HTTP status:", response.status_code)
    response.raise_for_status()

    data = response.json()

    print("status:", data["status"])
    print("number of submissions:", len(data["result"]))

    for submission in data["result"][:3]:
        print(submission)


if __name__ == "__main__":
    get_submissions("tourist")