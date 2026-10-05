from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/processed/pilot_contest_problem_events.parquet"
OUTPUT = ROOT / "data/processed/pilot_monthly_upsolve_features.parquet"


def main():
    print("=" * 70)
    print("BUILDING MONTHLY UPSOLVE FEATURES")
    print("=" * 70)

    df = pd.read_parquet(INPUT)

    # ---------------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------------

    required_columns = [
        "handle",
        "contest_id",
        "contest_end",
        "attempted_during_contest",
        "solved_during_contest",
        "strict_upsolve",
        "upsolve_24h",
        "upsolve_7d",
        "upsolve_30d",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    df["contest_end"] = pd.to_datetime(
        df["contest_end"],
        utc=True,
    )

    # ---------------------------------------------------------------
    # Month
    # ---------------------------------------------------------------

    # The contest month determines when the upsolve opportunity
    # occurred.
    df["month"] = (
        df["contest_end"]
        .dt.tz_convert(None)
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    # ---------------------------------------------------------------
    # Upsolve opportunity
    # ---------------------------------------------------------------

    # An upsolve opportunity exists when:
    #
    #   1. The user attempted the problem during the contest.
    #   2. The user did not solve it during the contest.
    #
    df["upsolve_opportunity"] = (
        df["attempted_during_contest"]
        & ~df["solved_during_contest"]
    )

    # ---------------------------------------------------------------
    # Ensure boolean feature columns are valid
    # ---------------------------------------------------------------

    boolean_columns = [
        "strict_upsolve",
        "upsolve_24h",
        "upsolve_7d",
        "upsolve_30d",
    ]

    for column in boolean_columns:
        df[column] = df[column].fillna(False).astype(bool)

    # These temporal upsolve measures are only meaningful for
    # actual upsolve opportunities.
    #
    # The upstream event builder already defines them, so we preserve
    # those definitions here rather than recomputing them.

    # ---------------------------------------------------------------
    # Aggregate opportunities by user-month
    # ---------------------------------------------------------------

    opportunities = (
        df[df["upsolve_opportunity"]]
        .groupby(
            ["handle", "month"],
            as_index=False,
        )
        .agg(
            upsolve_opportunities=(
                "upsolve_opportunity",
                "sum",
            ),
            strict_upsolves=(
                "strict_upsolve",
                "sum",
            ),
            upsolves_24h=(
                "upsolve_24h",
                "sum",
            ),
            upsolves_7d=(
                "upsolve_7d",
                "sum",
            ),
            upsolves_30d=(
                "upsolve_30d",
                "sum",
            ),
            contests_with_opportunity=(
                "contest_id",
                "nunique",
            ),
        )
    )

    # ---------------------------------------------------------------
    # Participated contests
    # ---------------------------------------------------------------

    participated = (
        df[df["attempted_during_contest"]]
        .groupby(
            ["handle", "month"],
            as_index=False,
        )
        .agg(
            contests_participated=(
                "contest_id",
                "nunique",
            )
        )
    )

    # ---------------------------------------------------------------
    # Combine
    # ---------------------------------------------------------------

    result = opportunities.merge(
        participated,
        on=["handle", "month"],
        how="left",
    )

    # ---------------------------------------------------------------
    # Rates
    # ---------------------------------------------------------------

    denominator = result["upsolve_opportunities"]

    result["upsolve_rate"] = (
        result["strict_upsolves"]
        / denominator
    )

    result["upsolve_rate_24h"] = (
        result["upsolves_24h"]
        / denominator
    )

    result["upsolve_rate_7d"] = (
        result["upsolves_7d"]
        / denominator
    )

    result["upsolve_rate_30d"] = (
        result["upsolves_30d"]
        / denominator
    )

    # ---------------------------------------------------------------
    # Ordering
    # ---------------------------------------------------------------

    result = (
        result
        .sort_values(["handle", "month"])
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------

    print()
    print(f"Rows: {len(result):,}")
    print(f"Users: {result['handle'].nunique():,}")
    print(f"Months: {result['month'].nunique():,}")

    print("\n=== UPSOLVE OPPORTUNITIES ===")
    print(
        result["upsolve_opportunities"].describe()
    )

    print("\n=== UPSOLVE RATE ===")
    print(
        result["upsolve_rate"].describe()
    )

    print("\n=== TEMPORAL UPSOLVE RATES ===")
    print(
        result[
            [
                "upsolve_rate",
                "upsolve_rate_24h",
                "upsolve_rate_7d",
                "upsolve_rate_30d",
            ]
        ].describe()
    )

    print("\n=== MONTHS WITH UPSOLVE OPPORTUNITIES ===")
    print(
        (
            result["upsolve_opportunities"] > 0
        ).mean()
    )

    print("\n=== STRICT UPSOLVE COUNTS ===")
    print(
        result["strict_upsolves"].describe()
    )

    print("\n=== SAMPLE ===")
    print(
        result.head(10).to_string(index=False)
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT,
        index=False,
    )

    print()
    print(f"Saved:\n{OUTPUT}")


if __name__ == "__main__":
    main()