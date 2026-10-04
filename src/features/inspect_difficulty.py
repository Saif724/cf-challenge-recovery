from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILE = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "difficulty_monthly.parquet"
)


def main():
    df = pd.read_parquet(FILE)

    cols = [
        "month",
        "attempted_rated_count",
        "accepted_rated_count",
        "median_accepted_rating",
        "mean_accepted_rating",
        "p75_accepted_rating",
        "p90_accepted_rating",
        "rating_at_month_start",
        "difficulty_gap",
    ]

    print("=== ALL MONTHS WITH ACCEPTED PROBLEMS ===")

    print(
        df[df["accepted_rated_count"] > 0][cols]
        .to_string(index=False)
    )

    print("\n=== GAP DISTRIBUTION BY ACCEPTED-PROBLEM COUNT ===")

    print(
        df[df["accepted_rated_count"] > 0]
        .groupby("accepted_rated_count")["difficulty_gap"]
        .agg(
            months="count",
            median="median",
            mean="mean",
            min="min",
            max="max",
        )
        .to_string()
    )

    print("\n=== PROBLEM-COUNT DISTRIBUTION ===")

    print(
        df["accepted_rated_count"]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()