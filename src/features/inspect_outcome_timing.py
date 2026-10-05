from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PANEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_monthly_development_panel.parquet"
)


def main():
    print("=" * 70)
    print("INSPECTING OUTCOME TIMING")
    print("=" * 70)

    df = pd.read_parquet(PANEL_FILE)

    df["month"] = pd.to_datetime(df["month"])

    # ------------------------------------------------------------
    # Focus on rated + practice months with a 3m outcome
    # ------------------------------------------------------------

    usable = df[
        df["rating_at_month_end"].notna()
        & (df["accepted_rated_count"] > 0)
        & df["rating_gain_3m"].notna()
    ].copy()

    print()
    print("Usable observations:", len(usable))
    print("Users:", usable["handle"].nunique())

    # ------------------------------------------------------------
    # Find next contest after month-end
    # ------------------------------------------------------------

    contests = df[
        df["active_contest_month"]
    ][
        [
            "handle",
            "month",
        ]
    ].copy()

    contests = contests.rename(
        columns={
            "month": "contest_month"
        }
    )

    contests = contests.sort_values(
        ["contest_month", "handle"]
    )

    usable = usable.sort_values(
        ["month", "handle"]
    )

    # Search for first contest month after current month.
    merged = pd.merge_asof(
        usable,
        contests,
        left_on="month",
        right_on="contest_month",
        by="handle",
        direction="forward",
        allow_exact_matches=False,
    )

    merged["months_to_next_contest"] = (
        (
            merged["contest_month"]
            .dt.year
            * 12
            + merged["contest_month"].dt.month
        )
        -
        (
            merged["month"].dt.year
            * 12
            + merged["month"].dt.month
        )
    )

    # ------------------------------------------------------------
    # Timing distribution
    # ------------------------------------------------------------

    print()
    print("=== MONTHS TO NEXT CONTEST ===")

    print(
        merged[
            "months_to_next_contest"
        ].describe()
    )

    print()
    print(
        merged[
            "months_to_next_contest"
        ].value_counts()
        .sort_index()
        .to_string()
    )

    # ------------------------------------------------------------
    # Outcome by distance
    # ------------------------------------------------------------

    print()
    print("=== OUTCOME BY DISTANCE TO NEXT CONTEST ===")

    summary = (
        merged
        .groupby(
            "months_to_next_contest"
        )
        .agg(
            observations=(
                "rating_gain_3m",
                "size",
            ),
            mean_rating_gain=(
                "rating_gain_3m",
                "mean",
            ),
            median_rating_gain=(
                "rating_gain_3m",
                "median",
            ),
            mean_gap_p75=(
                "difficulty_gap_p75",
                "mean",
            ),
        )
        .reset_index()
    )

    print(summary.to_string(index=False))

    # ------------------------------------------------------------
    # User-level distribution
    # ------------------------------------------------------------

    print()
    print("=== USABLE OBSERVATIONS PER USER ===")

    user_counts = (
        merged
        .groupby("handle")
        .size()
        .sort_values()
    )

    print(
        user_counts.describe()
    )

    print()
    print("Number of users by observation count:")

    print(
        user_counts
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ------------------------------------------------------------
    # Difficulty gap missingness
    # ------------------------------------------------------------

    print()
    print("=== DIFFICULTY GAP AVAILABILITY ===")

    for col in [
        "difficulty_gap_p50",
        "difficulty_gap_p75",
        "difficulty_gap_p90",
    ]:
        n = merged[col].notna().sum()

        print(
            f"{col}: {n} / {len(merged)}"
            f" = {100 * n / len(merged):.2f}%"
        )

    # ------------------------------------------------------------
    # Usable primary exposure
    # ------------------------------------------------------------

    primary = merged[
        merged["difficulty_gap_p75"].notna()
    ].copy()

    print()
    print("=== PRIMARY ANALYTICAL SAMPLE ===")

    print(
        "Rows:",
        len(primary),
    )

    print(
        "Users:",
        primary["handle"].nunique(),
    )

    print()
    print("Difficulty-gap P75:")

    print(
        primary[
            "difficulty_gap_p75"
        ].describe()
    )

    print()
    print("Rating gain 3m:")

    print(
        primary[
            "rating_gain_3m"
        ].describe()
    )

    print()
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()