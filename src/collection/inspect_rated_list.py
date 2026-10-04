from pathlib import Path
import json

from client import CodeforcesClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    client = CodeforcesClient()

    print("Fetching Codeforces rated user list...")
    users = client._request(
        "user.ratedList",
        {
            "activeOnly": "false",
            "includeRetired": "true",
        },
    )

    print()
    print("Number of users:", len(users))

    if users:
        print()
        print("Fields:")
        print(users[0].keys())

        print()
        print("First user:")
        print(json.dumps(users[0], indent=2))

    output = RAW_DIR / "rated_users.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

    print()
    print("Saved to:", output)


if __name__ == "__main__":
    main()