import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "tourist"
    / "user_info.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "users.parquet"
)


def load_user_info():
    with RAW_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize(user_info):
    rows = []

    for user in user_info:
        rows.append(
            {
                "handle": user.get("handle"),
                "first_name": user.get("firstName"),
                "last_name": user.get("lastName"),
                "country": user.get("country"),
                "city": user.get("city"),
                "organization": user.get("organization"),
                "rating": user.get("rating"),
                "max_rating": user.get("maxRating"),
                "rank": user.get("rank"),
                "max_rank": user.get("maxRank"),
                "contribution": user.get("contribution"),
                "friend_of_count": user.get("friendOfCount"),
                "registration_time_seconds": user.get(
                    "registrationTimeSeconds"
                ),
                "last_online_time_seconds": user.get(
                    "lastOnlineTimeSeconds"
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    user_info = load_user_info()

    print(f"Loaded {len(user_info)} raw user records.")

    df = normalize(user_info)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(OUTPUT_FILE, index=False)

    print("Saved normalized data to:")
    print(OUTPUT_FILE)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nUser:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()