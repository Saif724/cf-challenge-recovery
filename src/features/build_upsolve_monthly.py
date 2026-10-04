from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "contest_events.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "upsolve_monthly.parquet"
)


def main():
    events = pd.read_parquet(EVENTS_FILE)

    # ---------------------------------------------------------
    # Keep only problems actually attempted during the contest.
    # ---------------------------------------------------------

    events = events[
        events["attempted_during_contest"] > 0
    ].copy()

    # ---------------------------------------------------------
    # Month of the original contest.
    #
    # first_contest_submission is guaranteed to occur during
    # the official contest for these events.
    # ---------------------------------------------------------

    events["contest_month"] = (
        events["first_contest_submission"]
        .dt.strftime("%Y-%m")
    )

    # ---------------------------------------------------------
    # Monthly contest/upsolving counts
    # ---------------------------------------------------------

    monthly = (
        events
        .groupby(
            ["handle", "contest_month"],
            as_index=False,
        )
        .agg(
            attempted_contest_problem_count=(
                "problem_index",
                "count",
            ),
            attempted_but_unsolved_count=(
                "attempted_but_unsolved",
                "sum",
            ),
            strict_upsolve_count=(
                "strict_upsolve",
                "sum",
            ),
        )
    )

    # ---------------------------------------------------------
    # Delay-based upsolve features
    # ---------------------------------------------------------

    strict = events[
        events["strict_upsolve"]
    ].copy()

    if not strict.empty:

        strict["upsolve_within_1d"] = (
            strict["solve_delay_hours"] <= 24
        )

        strict["upsolve_within_3d"] = (
            strict["solve_delay_hours"] <= 72
        )

        strict["upsolve_within_7d"] = (
            strict["solve_delay_hours"] <= 168
        )

        delay_features = (
            strict
            .groupby(
                ["handle", "contest_month"],
                as_index=False,
            )
            .agg(
                upsolve_1d_count=(
                    "upsolve_within_1d",
                    "sum",
                ),
                upsolve_3d_count=(
                    "upsolve_within_3d",
                    "sum",
                ),
                upsolve_7d_count=(
                    "upsolve_within_7d",
                    "sum",
                ),
                median_upsolve_delay_hours=(
                    "solve_delay_hours",
                    "median",
                ),
            )
        )

        monthly = monthly.merge(
            delay_features,
            on=[
                "handle",
                "contest_month",
            ],
            how="left",
        )

    else:
        monthly["upsolve_1d_count"] = 0
        monthly["upsolve_3d_count"] = 0
        monthly["upsolve_7d_count"] = 0
        monthly["median_upsolve_delay_hours"] = np.nan

    # ---------------------------------------------------------
    # Fill count columns.
    #
    # A month with no strict upsolve has zero count.
    # ---------------------------------------------------------

    count_columns = [
        "upsolve_1d_count",
        "upsolve_3d_count",
        "upsolve_7d_count",
    ]

    for column in count_columns:
        monthly[column] = (
            monthly[column]
            .fillna(0)
            .astype(int)
        )

    # ---------------------------------------------------------
    # Primary upsolve rate
    #
    # IMPORTANT:
    # No attempted-but-unsolved problems means there was no
    # opportunity to upsolve.
    #
    # Therefore use NaN, NOT 0.
    # ---------------------------------------------------------

    monthly["upsolve_rate"] = np.where(
        monthly["attempted_but_unsolved_count"] > 0,
        (
            monthly["strict_upsolve_count"]
            / monthly["attempted_but_unsolved_count"]
        ),
        np.nan,
    )

    monthly = monthly.sort_values(
        ["handle", "contest_month"]
    )

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

    print("=== UPSOLVE MONTHLY ===")

    print(f"Rows: {len(monthly):,}")
    print(
        f"Users: {monthly['handle'].nunique():,}"
    )
    print(
        f"Months: {monthly['contest_month'].nunique():,}"
    )

    print("\n=== TOTALS ===")

    print(
        "Attempted contest problems:",
        int(
            monthly[
                "attempted_contest_problem_count"
            ].sum()
        ),
    )

    print(
        "Attempted but unsolved:",
        int(
            monthly[
                "attempted_but_unsolved_count"
            ].sum()
        ),
    )

    print(
        "Strict upsolves:",
        int(
            monthly[
                "strict_upsolve_count"
            ].sum()
        ),
    )

    print("\n=== UPSOLVE RATE ===")

    print(
        monthly["upsolve_rate"]
        .describe()
        .to_string()
    )

    print("\n=== MONTHLY SAMPLE ===")

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