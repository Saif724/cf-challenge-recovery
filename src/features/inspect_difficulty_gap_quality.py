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
    print("INSPECTING DIFFICULTY GAP QUALITY")
    print("=" * 70)

    df = pd.read_parquet(PANEL_FILE)

    df["month"] = pd.to_datetime(df["month"])

    # Primary analytical sample
    sample = df[
        df["rating_at_month_end"].notna()
        & (df["accepted_rated_count"] > 0)
        & df["rating_gain_3m"].notna()
        & df["difficulty_gap_p75"].notna()
    ].copy()

    print()
    print("=== PRIMARY SAMPLE ===")
    print("Rows:", len(sample))
    print("Users:", sample["handle"].nunique())

    # ------------------------------------------------------------
    # Basic relationship between exposure and practice volume
    # ------------------------------------------------------------

    print()
    print("=== PRACTICE VOLUME VS DIFFICULTY GAP ===")

    print(
        sample[
            [
                "accepted_rated_count",
                "difficulty_gap_p50",
                "difficulty_gap_p75",
                "difficulty_gap_p90",
            ]
        ].corr().to_string()
    )

    # ------------------------------------------------------------
    # Extreme observations
    # ------------------------------------------------------------

    print()
    print("=== 20 LOWEST P75 GAPS ===")

    cols = [
        "handle",
        "month",
        "rating_at_month_end",
        "accepted_rated_count",
        "practice_p50",
        "practice_p75",
        "practice_p90",
        "difficulty_gap_p50",
        "difficulty_gap_p75",
        "difficulty_gap_p90",
        "rating_gain_3m",
    ]

    print(
        sample
        .sort_values("difficulty_gap_p75")
        [cols]
        .head(20)
        .to_string(index=False)
    )

    print()
    print("=== 20 HIGHEST P75 GAPS ===")

    print(
        sample
        .sort_values("difficulty_gap_p75", ascending=False)
        [cols]
        .head(20)
        .to_string(index=False)
    )

    # ------------------------------------------------------------
    # Distribution by practice volume
    # ------------------------------------------------------------

    print()
    print("=== GAP DISTRIBUTION BY PRACTICE VOLUME ===")

    sample["practice_volume_group"] = pd.cut(
        sample["accepted_rated_count"],
        bins=[0, 1, 2, 5, 10, 20, 50, float("inf")],
        labels=[
            "1",
            "2",
            "3-5",
            "6-10",
            "11-20",
            "21-50",
            "51+",
        ],
    )

    volume_summary = (
        sample
        .groupby(
            "practice_volume_group",
            observed=True,
        )
        .agg(
            observations=("handle", "size"),
            users=("handle", "nunique"),
            median_gap=("difficulty_gap_p75", "median"),
            mean_gap=("difficulty_gap_p75", "mean"),
            p25_gap=("difficulty_gap_p75", lambda x: x.quantile(0.25)),
            p75_gap=("difficulty_gap_p75", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )

    print(
        volume_summary.to_string(index=False)
    )

    # ------------------------------------------------------------
    # Extreme-value counts
    # ------------------------------------------------------------

    print()
    print("=== EXTREME GAP COUNTS ===")

    for threshold in [500, 750, 1000, 1500]:
        high = (
            sample["difficulty_gap_p75"] >= threshold
        ).sum()

        low = (
            sample["difficulty_gap_p75"] <= -threshold
        ).sum()

        print(
            f"|gap| >= {threshold:4d}: "
            f"high={high:3d}, low={low:3d}"
        )

    # ------------------------------------------------------------
    # Rating and practice volume for extreme observations
    # ------------------------------------------------------------

    print()
    print("=== EXTREME OBSERVATIONS SUMMARY ===")

    extreme = sample[
        sample["difficulty_gap_p75"].abs() >= 750
    ]

    print(
        extreme[
            [
                "rating_at_month_end",
                "accepted_rated_count",
                "practice_p75",
                "difficulty_gap_p75",
                "rating_gain_3m",
            ]
        ].describe()
    )

    # ------------------------------------------------------------
    # Relationship with outcome
    # ------------------------------------------------------------

    print()
    print("=== EXPOSURE / OUTCOME CORRELATION ===")

    print(
        sample[
            [
                "difficulty_gap_p50",
                "difficulty_gap_p75",
                "difficulty_gap_p90",
                "rating_gain_3m",
            ]
        ].corr()["rating_gain_3m"].to_string()
    )

    print()
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()