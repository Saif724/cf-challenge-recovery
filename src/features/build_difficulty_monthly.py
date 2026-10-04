from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRACTICE_FILE = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "practice_monthly.parquet"
)

RATING_FILE = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "rating_monthly.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "difficulty_monthly.parquet"
)


def main():
    practice = pd.read_parquet(PRACTICE_FILE)
    ratings = pd.read_parquet(RATING_FILE)

    # Keep only the variables needed for the merge.
    ratings = ratings[
        [
            "handle",
            "month",
            "rating_at_month_start",
        ]
    ]

    monthly = practice.merge(
        ratings,
        on=["handle", "month"],
        how="left",
        validate="one_to_one",
    )

    # Primary difficulty-gap measure.
    #
    # Positive value:
    #   accepted problems were harder than current rating.
    #
    # Negative value:
    #   accepted problems were easier than current rating.
    #
    # NaN:
    #   no accepted rated problem in that month.
    monthly["difficulty_gap"] = (
        monthly["median_accepted_rating"]
        - monthly["rating_at_month_start"]
    )

    # Secondary versions for robustness analysis.
    monthly["mean_difficulty_gap"] = (
        monthly["mean_accepted_rating"]
        - monthly["rating_at_month_start"]
    )

    monthly["p75_difficulty_gap"] = (
        monthly["p75_accepted_rating"]
        - monthly["rating_at_month_start"]
    )

    monthly["p90_difficulty_gap"] = (
        monthly["p90_accepted_rating"]
        - monthly["rating_at_month_start"]
    )

    # Binary challenge thresholds.
    monthly["gap_ge_100"] = (
        monthly["difficulty_gap"] >= 100
    )

    monthly["gap_ge_200"] = (
        monthly["difficulty_gap"] >= 200
    )

    monthly["gap_ge_300"] = (
        monthly["difficulty_gap"] >= 300
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    monthly.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("=== DIFFICULTY MONTHLY ===")

    print(f"Rows: {len(monthly):,}")
    print(
        f"Users: {monthly['handle'].nunique():,}"
    )
    print(
        f"Months: {monthly['month'].nunique():,}"
    )

    print("\n=== MISSING ===")

    print(
        "Missing rating:",
        int(
            monthly["rating_at_month_start"]
            .isna()
            .sum()
        ),
    )

    print(
        "Missing difficulty gap:",
        int(
            monthly["difficulty_gap"]
            .isna()
            .sum()
        ),
    )

    print("\n=== DIFFICULTY GAP ===")

    print(
        monthly["difficulty_gap"]
        .describe()
        .to_string()
    )

    print("\n=== THRESHOLDS ===")

    valid_gap = monthly["difficulty_gap"].dropna()

    print(
        "Gap >= 100:",
        int((valid_gap >= 100).sum()),
        f"({(valid_gap >= 100).mean():.2%})",
    )

    print(
        "Gap >= 200:",
        int((valid_gap >= 200).sum()),
        f"({(valid_gap >= 200).mean():.2%})",
    )

    print(
        "Gap >= 300:",
        int((valid_gap >= 300).sum()),
        f"({(valid_gap >= 300).mean():.2%})",
    )

    print("\n=== SAMPLE ===")

    print(
        monthly[
            [
                "handle",
                "month",
                "median_accepted_rating",
                "rating_at_month_start",
                "difficulty_gap",
                "gap_ge_100",
                "gap_ge_200",
                "gap_ge_300",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

    print(
        f"\nSaved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()