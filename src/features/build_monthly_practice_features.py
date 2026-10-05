from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRACTICE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_practice_problems.parquet"
)

RATING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_rating_history.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_monthly_practice_features.parquet"
)


def main():
    practice = pd.read_parquet(PRACTICE_FILE)
    ratings = pd.read_parquet(RATING_FILE)

    print("=" * 70)
    print("BUILDING MONTHLY PRACTICE FEATURES")
    print("=" * 70)

    # ------------------------------------------------------------
    # Normalize timestamps to timezone-naive UTC-like timestamps.
    # ------------------------------------------------------------

    practice["submission_time"] = pd.to_datetime(
        practice["submission_time"],
        utc=True,
    ).dt.tz_convert(None)

    ratings["contest_time"] = pd.to_datetime(
        ratings["contest_time"],
        utc=True,
    ).dt.tz_convert(None)

    # ------------------------------------------------------------
    # Practice month
    # ------------------------------------------------------------

    practice["month"] = (
        practice["submission_time"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    # ------------------------------------------------------------
    # Rating events
    # ------------------------------------------------------------

    rating_events = (
        ratings[
            [
                "handle",
                "contest_time",
                "new_rating",
            ]
        ]
        .sort_values(
            [
                "handle",
                "contest_time",
            ]
        )
        .copy()
    )

    # ------------------------------------------------------------
    # User-month combinations
    # ------------------------------------------------------------

    user_months = (
        practice[
            [
                "handle",
                "month",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "handle",
                "month",
            ]
        )
        .copy()
    )

    # ------------------------------------------------------------
    # Latest official rating strictly BEFORE month start.
    # ------------------------------------------------------------

    rating_rows = []

    for handle, months in user_months.groupby(
        "handle",
        sort=False,
    ):
        months = months.copy()

        user_ratings = rating_events[
            rating_events["handle"] == handle
        ][
            [
                "contest_time",
                "new_rating",
            ]
        ].sort_values("contest_time")

        if user_ratings.empty:
            months["rating_before_month"] = pd.NA

        else:
            rating_times = (
                user_ratings["contest_time"]
                .tolist()
            )

            rating_values = (
                user_ratings["new_rating"]
                .tolist()
            )

            ratings_for_month = []

            for month_start in months["month"]:
                previous = [
                    i
                    for i, rating_time
                    in enumerate(rating_times)
                    if rating_time < month_start
                ]

                if previous:
                    ratings_for_month.append(
                        rating_values[previous[-1]]
                    )
                else:
                    ratings_for_month.append(
                        pd.NA
                    )

            months["rating_before_month"] = (
                ratings_for_month
            )

        rating_rows.append(months)

    monthly_rating = pd.concat(
        rating_rows,
        ignore_index=True,
    )

    # ------------------------------------------------------------
    # Attach monthly rating to practice problems
    # ------------------------------------------------------------

    practice = practice.merge(
        monthly_rating,
        on=[
            "handle",
            "month",
        ],
        how="left",
    )

    # ------------------------------------------------------------
    # Difficulty gap
    # ------------------------------------------------------------

    practice["difficulty_gap"] = (
        practice["problem_rating"]
        - pd.to_numeric(
            practice["rating_before_month"],
            errors="coerce",
        )
    )

    # ------------------------------------------------------------
    # Aggregate practice difficulty
    # ------------------------------------------------------------

    grouped = practice.groupby(
        [
            "handle",
            "month",
            "practice_definition",
        ]
    )

    features = (
        grouped["problem_rating"]
        .agg(
            accepted_rated_count="count",
            practice_p50="median",
            practice_p75=lambda x: x.quantile(0.75),
            practice_p90=lambda x: x.quantile(0.90),
        )
        .reset_index()
    )

    gap_features = (
        grouped["difficulty_gap"]
        .agg(
            difficulty_gap_p50="median",
            difficulty_gap_p75=lambda x: x.quantile(0.75),
            difficulty_gap_p90=lambda x: x.quantile(0.90),
        )
        .reset_index()
    )

    features = features.merge(
        gap_features,
        on=[
            "handle",
            "month",
            "practice_definition",
        ],
        how="left",
    )

    # ------------------------------------------------------------
    # Attach rating state
    # ------------------------------------------------------------

    features = features.merge(
        monthly_rating,
        on=[
            "handle",
            "month",
        ],
        how="left",
    )

    # ------------------------------------------------------------
    # Sort and save
    # ------------------------------------------------------------

    features = features.sort_values(
        [
            "handle",
            "month",
            "practice_definition",
        ]
    )

    features.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    print()
    print("Rows:", len(features))
    print("Users:", features["handle"].nunique())
    print("Months:", features["month"].nunique())

    print()
    print("=== ROWS BY DEFINITION ===")
    print(
        features["practice_definition"]
        .value_counts()
    )

    print()
    print("=== ACCEPTED PRACTICE COUNT ===")
    print(
        features.groupby("practice_definition")[
            "accepted_rated_count"
        ].describe()
    )

    print()
    print("=== DIFFICULTY GAP SUMMARY ===")
    print(
        features.groupby("practice_definition")[
            [
                "difficulty_gap_p50",
                "difficulty_gap_p75",
                "difficulty_gap_p90",
            ]
        ].agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max",
            ]
        )
    )

    print()
    print("=== MISSING MONTHLY RATING ===")
    print(
        features["rating_before_month"]
        .isna()
        .value_counts()
    )

    print()
    print("=== P75 GAP THRESHOLDS ===")

    for threshold in [100, 200, 300, 500]:
        print(
            f"\nP75 gap >= {threshold}:"
        )

        print(
            features.assign(
                hard=(
                    features["difficulty_gap_p75"]
                    >= threshold
                )
            )
            .groupby("practice_definition")["hard"]
            .mean()
        )

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()