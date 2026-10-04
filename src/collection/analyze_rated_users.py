from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT = PROJECT_ROOT / "data" / "raw" / "rated_users.json"


def main():
    print("Loading rated users...")

    with INPUT.open("r", encoding="utf-8") as f:
        users = json.load(f)

    df = pd.DataFrame(users)

    print()
    print("=== BASIC ===")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    print()
    print("=== CURRENT RATING ===")
    print(df["rating"].describe())

    print()
    print("=== MAX RATING ===")
    print(df["maxRating"].describe())

    print()
    print("=== REGISTRATION ===")

    registration = pd.to_datetime(
        df["registrationTimeSeconds"],
        unit="s",
        utc=True,
    )

    print(registration.describe())

    df["registration_date"] = registration.dt.date
    df["registration_year"] = registration.dt.year

    print()
    print("Users by registration year:")
    print(df["registration_year"].value_counts().sort_index().tail(15))

    print()
    print("=== RECENT REGISTRATIONS ===")

    recent = df[df["registration_year"].between(2019, 2026)]

    print(
        recent.groupby("registration_year")
        .size()
        .sort_index()
    )

    print()
    print("=== CURRENTLY ACTIVE-LOOKING USERS ===")

    last_online = pd.to_datetime(
        df["lastOnlineTimeSeconds"],
        unit="s",
        utc=True,
    )

    df["last_online"] = last_online

    cutoff = last_online.max() - pd.Timedelta(days=365)

    print("Cutoff:", cutoff)
    print("Online within 1 year:", (last_online >= cutoff).sum())

    print()
    print("=== RATING DISTRIBUTION ===")
    print(
        df["rating"]
        .value_counts(bins=[0, 1199, 1399, 1599, 1799, 1999, 2399, 2799, 3199, 10000])
        .sort_index()
    )


if __name__ == "__main__":
    main()