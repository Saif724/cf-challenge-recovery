import pandas as pd
import numpy as np


INPUT = "data/processed/pilot_monthly_development_panel.parquet"


def describe_exposure(df, label):
    x = df["difficulty_gap_p75"].dropna()

    print()
    print("=" * 80)
    print(label)
    print("=" * 80)

    print(f"Observations : {len(x)}")
    print(f"Users        : {df.loc[x.index, 'handle'].nunique()}")

    if len(x) == 0:
        print("No observations.")
        return

    q = x.quantile(
        [
            0.00,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            1.00,
        ]
    )

    print()
    print("DISTRIBUTION")
    print("-" * 80)

    print(f"Mean   : {x.mean():.2f}")
    print(f"SD     : {x.std():.2f}")
    print(f"Min    : {q.loc[0.00]:.2f}")
    print(f"P05    : {q.loc[0.05]:.2f}")
    print(f"P10    : {q.loc[0.10]:.2f}")
    print(f"P25    : {q.loc[0.25]:.2f}")
    print(f"Median : {q.loc[0.50]:.2f}")
    print(f"P75    : {q.loc[0.75]:.2f}")
    print(f"P90    : {q.loc[0.90]:.2f}")
    print(f"P95    : {q.loc[0.95]:.2f}")
    print(f"P99    : {q.loc[0.99]:.2f}")
    print(f"Max    : {q.loc[1.00]:.2f}")

    print()
    print("SIGN")
    print("-" * 80)

    print(f"Negative gap : {(x < 0).sum():>6} ({(x < 0).mean():.2%})")
    print(f"Zero gap     : {(x == 0).sum():>6} ({(x == 0).mean():.2%})")
    print(f"Positive gap : {(x > 0).sum():>6} ({(x > 0).mean():.2%})")

    print()
    print("EXTREME THRESHOLDS")
    print("-" * 80)

    for threshold in [250, 500, 750, 1000, 1500]:
        high = (x >= threshold).sum()
        low = (x <= -threshold).sum()

        print(
            f"|gap| >= {threshold:4d}: "
            f"high={high:4d}, "
            f"low={low:4d}, "
            f"total={high + low:4d}"
        )


def main():
    print("=" * 80)
    print("DIFFICULTY GAP EXPOSURE QUALITY")
    print("=" * 80)

    df = pd.read_parquet(INPUT)

    print(f"Panel rows : {len(df)}")
    print(f"Users      : {df['handle'].nunique()}")

    # ------------------------------------------------------------------
    # Primary analytical sample
    # ------------------------------------------------------------------

    primary = df[
        df["rating_at_month_end"].notna()
        & (df["accepted_rated_count"] > 0)
        & df["rating_gain_3m"].notna()
        & df["difficulty_gap_p75"].notna()
    ].copy()

    describe_exposure(primary, "PRIMARY SAMPLE")

    # ------------------------------------------------------------------
    # Practice-volume sensitivity
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("PRACTICE-VOLUME SENSITIVITY")
    print("=" * 80)

    for threshold in [1, 3, 5, 10]:
        subset = primary[
            primary["accepted_rated_count"] >= threshold
        ].copy()

        print()
        print(f"n >= {threshold}")
        print("-" * 40)

        print(f"Observations : {len(subset)}")
        print(f"Users        : {subset['handle'].nunique()}")

        if len(subset) > 0:
            x = subset["difficulty_gap_p75"]

            print(f"Mean         : {x.mean():.2f}")
            print(f"Median       : {x.median():.2f}")
            print(f"SD           : {x.std():.2f}")
            print(f"P10          : {x.quantile(.10):.2f}")
            print(f"P90          : {x.quantile(.90):.2f}")
            print(f"Min          : {x.min():.2f}")
            print(f"Max          : {x.max():.2f}")

    # ------------------------------------------------------------------
    # Exposure vs outcome descriptive correlation
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("RAW EXPOSURE / OUTCOME ASSOCIATION")
    print("=" * 80)

    print(
        "Pearson correlation:",
        primary["difficulty_gap_p75"].corr(primary["rating_gain_3m"])
    )

    print(
        "Spearman correlation:",
        primary["difficulty_gap_p75"].corr(
            primary["rating_gain_3m"],
            method="spearman",
        )
    )

    # ------------------------------------------------------------------
    # Correlation with baseline rating
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("EXPOSURE / BASELINE RATING")
    print("=" * 80)

    print(
        "difficulty_gap_p75 vs rating_at_month_end:",
        primary["difficulty_gap_p75"].corr(
            primary["rating_at_month_end"]
        )
    )

    # ------------------------------------------------------------------
    # Practice volume relationship
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("EXPOSURE / PRACTICE VOLUME")
    print("=" * 80)

    print(
        "difficulty_gap_p75 vs accepted_rated_count:",
        primary["difficulty_gap_p75"].corr(
            primary["accepted_rated_count"]
        )
    )


if __name__ == "__main__":
    main()