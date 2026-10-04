from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RATINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ratings.parquet"
)

PRACTICE_FILE = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "practice_monthly.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "rating_monthly.parquet"
)


def main():
    ratings = pd.read_parquet(RATINGS_FILE)
    practice = pd.read_parquet(PRACTICE_FILE)

    # ---------------------------------------------------------
    # Rating update timestamp
    # ---------------------------------------------------------

    ratings["rating_time"] = pd.to_datetime(
        ratings["rating_update_time_seconds"],
        unit="s",
        utc=True,
    ).astype("datetime64[ns, UTC]")

    ratings = ratings.sort_values(
        ["rating_time", "handle"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Practice months
    # ---------------------------------------------------------

    months = practice[
        [
            "handle",
            "month",
        ]
    ].drop_duplicates()

    months["month_start"] = pd.to_datetime(
        months["month"] + "-01",
        utc=True,
    ).astype("datetime64[ns, UTC]")

    # IMPORTANT:
    # merge_asof requires the actual merge key to be globally
    # sorted. Therefore sort by month_start FIRST.
    months = months.sort_values(
        ["month_start", "handle"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Most recent official rating strictly BEFORE
    # the beginning of the practice month.
    #
    # This prevents rating updates occurring during the month
    # from leaking into that month's predictors.
    # ---------------------------------------------------------

    rating_lookup = ratings[
        [
            "handle",
            "rating_time",
            "new_rating",
        ]
    ].copy()

    rating_lookup = rating_lookup.sort_values(
        ["rating_time", "handle"]
    ).reset_index(drop=True)

    monthly = pd.merge_asof(
        months,
        rating_lookup,
        left_on="month_start",
        right_on="rating_time",
        by="handle",
        direction="backward",
        allow_exact_matches=False,
    )

    monthly = monthly.rename(
        columns={
            "new_rating": "rating_at_month_start",
        }
    )

    # ---------------------------------------------------------
    # Months since first rated contest
    #
    # Use the first rating-update timestamp as the beginning
    # of the user's Codeforces rated history.
    # ---------------------------------------------------------

    first_rating = (
        ratings
        .groupby("handle", as_index=False)
        .agg(
            first_rating_time=(
                "rating_time",
                "min",
            )
        )
    )

    monthly = monthly.merge(
        first_rating,
        on="handle",
        how="left",
    )

    monthly["months_since_first_rating"] = (
        (
            monthly["month_start"].dt.year
            - monthly["first_rating_time"].dt.year
        )
        * 12
        +
        (
            monthly["month_start"].dt.month
            - monthly["first_rating_time"].dt.month
        )
    )

    # Restore a convenient user/month ordering.
    monthly = monthly.sort_values(
        ["handle", "month_start"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    monthly.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    print("=== RATING MONTHLY ===")

    print(f"Rows: {len(monthly):,}")
    print(
        f"Users: {monthly['handle'].nunique():,}"
    )
    print(
        f"Months: {monthly['month'].nunique():,}"
    )

    print("\n=== MISSING RATINGS ===")

    print(
        "Missing rating:",
        int(
            monthly["rating_at_month_start"]
            .isna()
            .sum()
        ),
    )

    print("\n=== RATING DISTRIBUTION ===")

    print(
        monthly["rating_at_month_start"]
        .describe()
        .to_string()
    )

    print("\n=== SAMPLE ===")

    print(
        monthly.head(30).to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()