import pandas as pd
import numpy as np


INPUT = "data/processed/pilot_monthly_development_panel.parquet"


def main():
    print("=" * 80)
    print("WITHIN-USER DIFFICULTY GAP VARIATION")
    print("=" * 80)

    df = pd.read_parquet(INPUT)

    # Same primary definition used elsewhere.
    df = df[
        df["rating_at_month_end"].notna()
        & (df["accepted_rated_count"] > 0)
        & df["rating_gain_3m"].notna()
        & df["difficulty_gap_p75"].notna()
    ].copy()

    print()
    print("PRIMARY SAMPLE")
    print("-" * 80)

    print(f"Observations : {len(df)}")
    print(f"Users        : {df['handle'].nunique()}")

    # ------------------------------------------------------------------
    # Per-user statistics
    # ------------------------------------------------------------------

    user_stats = (
        df.groupby("handle")
        .agg(
            observations=("difficulty_gap_p75", "size"),
            gap_mean=("difficulty_gap_p75", "mean"),
            gap_sd=("difficulty_gap_p75", "std"),
            gap_min=("difficulty_gap_p75", "min"),
            gap_max=("difficulty_gap_p75", "max"),
            gap_median=("difficulty_gap_p75", "median"),
            rating_mean=("rating_at_month_end", "mean"),
        )
        .reset_index()
    )

    user_stats["gap_range"] = (
        user_stats["gap_max"] - user_stats["gap_min"]
    )

    print()
    print("=" * 80)
    print("OBSERVATIONS PER USER")
    print("=" * 80)

    print(user_stats["observations"].describe())

    print()
    print("Users by number of usable observations:")
    print(
        user_stats["observations"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ------------------------------------------------------------------
    # Within-user SD
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("WITHIN-USER STANDARD DEVIATION")
    print("=" * 80)

    valid_sd = user_stats["gap_sd"].dropna()

    print(f"Users with >= 2 observations : {len(valid_sd)}")
    print(f"Mean within-user SD          : {valid_sd.mean():.2f}")
    print(f"Median within-user SD        : {valid_sd.median():.2f}")
    print(f"P25                          : {valid_sd.quantile(.25):.2f}")
    print(f"P75                          : {valid_sd.quantile(.75):.2f}")
    print(f"P90                          : {valid_sd.quantile(.90):.2f}")
    print(f"Max                          : {valid_sd.max():.2f}")

    # ------------------------------------------------------------------
    # Within-user range
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("WITHIN-USER RANGE")
    print("=" * 80)

    ranges = user_stats["gap_range"].dropna()

    print(f"Mean range   : {ranges.mean():.2f}")
    print(f"Median range : {ranges.median():.2f}")
    print(f"P25          : {ranges.quantile(.25):.2f}")
    print(f"P75          : {ranges.quantile(.75):.2f}")
    print(f"P90          : {ranges.quantile(.90):.2f}")
    print(f"Max          : {ranges.max():.2f}")

    # ------------------------------------------------------------------
    # Users with meaningful variation
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("USERS WITH MEANINGFUL WITHIN-USER VARIATION")
    print("=" * 80)

    for threshold in [50, 100, 200, 300, 500, 750]:
        count = (ranges >= threshold).sum()

        print(
            f"Range >= {threshold:4d}: "
            f"{count:3d} users "
            f"({count / len(ranges):.2%})"
        )

    # ------------------------------------------------------------------
    # Between vs within variation
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("BETWEEN / WITHIN VARIANCE")
    print("=" * 80)

    overall_mean = df["difficulty_gap_p75"].mean()

    user_means = (
        df.groupby("handle")["difficulty_gap_p75"]
        .mean()
    )

    between_variance = (
        user_means - overall_mean
    ).pow(2).mean()

    within_variance = (
        df["difficulty_gap_p75"]
        - df.groupby("handle")["difficulty_gap_p75"]
        .transform("mean")
    ).pow(2).mean()

    total_variance = df["difficulty_gap_p75"].var()

    print(f"Overall variance    : {total_variance:.2f}")
    print(f"Between-user var    : {between_variance:.2f}")
    print(f"Within-user var     : {within_variance:.2f}")

    if total_variance > 0:
        print()
        print(
            "Approx. within-user share:",
            within_variance / total_variance
        )

    # ------------------------------------------------------------------
    # Users with almost no variation
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOW-VARIATION USERS")
    print("=" * 80)

    for threshold in [25, 50, 100]:
        count = (ranges < threshold).sum()

        print(
            f"Range < {threshold:3d}: "
            f"{count:3d} users "
            f"({count / len(ranges):.2%})"
        )

    # ------------------------------------------------------------------
    # Top users by variation
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("TOP 20 USERS BY WITHIN-USER RANGE")
    print("=" * 80)

    print(
        user_stats[
            [
                "handle",
                "observations",
                "gap_mean",
                "gap_sd",
                "gap_min",
                "gap_max",
                "gap_range",
            ]
        ]
        .sort_values("gap_range", ascending=False)
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()