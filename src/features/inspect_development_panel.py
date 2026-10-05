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
    print("INSPECTING MONTHLY DEVELOPMENT PANEL")
    print("=" * 70)

    df = pd.read_parquet(PANEL_FILE)

    # ------------------------------------------------------------
    # Basic
    # ------------------------------------------------------------

    print()
    print("=== BASIC ===")
    print("Rows:", len(df))
    print("Users:", df["handle"].nunique())
    print("Months:", df["month"].nunique())

    # ------------------------------------------------------------
    # Key intersections
    # ------------------------------------------------------------

    rated = df["rating_at_month_end"].notna()
    practice = df["accepted_rated_count"] > 0
    upsolve = df["upsolve_opportunities"] > 0
    contest = df["active_contest_month"]

    outcome_1 = df["rating_gain_1m"].notna()
    outcome_3 = df["rating_gain_3m"].notna()
    outcome_6 = df["rating_gain_6m"].notna()

    print()
    print("=== KEY INTERSECTIONS ===")

    checks = {
        "rated months": rated,
        "practice months": practice,
        "rated + practice": rated & practice,
        "rated + practice + 1m outcome":
            rated & practice & outcome_1,
        "rated + practice + 3m outcome":
            rated & practice & outcome_3,
        "rated + practice + 6m outcome":
            rated & practice & outcome_6,
        "rated + upsolve opportunity":
            rated & upsolve,
        "rated + upsolve + 3m outcome":
            rated & upsolve & outcome_3,
        "rated + contest":
            rated & contest,
        "rated + contest + 3m outcome":
            rated & contest & outcome_3,
    }

    for name, mask in checks.items():
        print(
            f"{name:40s}: {mask.sum():5d}"
        )

    # ------------------------------------------------------------
    # Coverage among rated months
    # ------------------------------------------------------------

    print()
    print("=== OUTCOME COVERAGE AMONG RATED MONTHS ===")

    rated_rows = rated.sum()

    for horizon in [1, 3, 6]:
        col = f"rating_gain_{horizon}m"
        n = df[col].notna().sum()

        print(
            f"{horizon}m: {n:5d} / {rated_rows:5d}"
            f" = {100 * n / rated_rows:.2f}%"
        )

    # ------------------------------------------------------------
    # Coverage among practice months
    # ------------------------------------------------------------

    print()
    print("=== OUTCOME COVERAGE AMONG RATED + PRACTICE MONTHS ===")

    rated_practice = rated & practice
    n_rated_practice = rated_practice.sum()

    print(
        "Rated + practice months:",
        n_rated_practice,
    )

    for horizon in [1, 3, 6]:
        col = f"rating_gain_{horizon}m"
        n = (
            rated_practice
            & df[col].notna()
        ).sum()

        print(
            f"{horizon}m: {n:5d} / {n_rated_practice:5d}"
            f" = {100 * n / n_rated_practice:.2f}%"
        )

    # ------------------------------------------------------------
    # Coverage by practice status
    # ------------------------------------------------------------

    print()
    print("=== 3M OUTCOME BY PRACTICE STATUS ===")

    for label, mask in [
        ("practice month", practice),
        ("non-practice month", ~practice),
    ]:
        denominator = (
            rated & mask
        ).sum()

        numerator = (
            rated
            & mask
            & outcome_3
        ).sum()

        print(
            f"{label:25s}:"
            f" {numerator:5d} / {denominator:5d}"
            f" = {100 * numerator / denominator:.2f}%"
            if denominator
            else
            f"{label:25s}: no observations"
        )

    # ------------------------------------------------------------
    # 3M outcome by current contest month
    # ------------------------------------------------------------

    print()
    print("=== 3M OUTCOME BY CURRENT CONTEST STATUS ===")

    for label, mask in [
        ("contest month", contest),
        ("non-contest month", ~contest),
    ]:
        denominator = (
            rated & mask
        ).sum()

        numerator = (
            rated
            & mask
            & outcome_3
        ).sum()

        print(
            f"{label:25s}:"
            f" {numerator:5d} / {denominator:5d}"
            f" = {100 * numerator / denominator:.2f}%"
            if denominator
            else
            f"{label:25s}: no observations"
        )

    # ------------------------------------------------------------
    # Outcome distributions
    # ------------------------------------------------------------

    print()
    print("=== OUTCOME DISTRIBUTIONS ===")

    for horizon in [1, 3, 6]:
        col = f"rating_gain_{horizon}m"

        print()
        print(col)

        print(
            df[col]
            .dropna()
            .describe(
                percentiles=[
                    0.01,
                    0.05,
                    0.25,
                    0.50,
                    0.75,
                    0.95,
                    0.99,
                ]
            )
        )

    # ------------------------------------------------------------
    # Practice exposure distributions in usable 3m observations
    # ------------------------------------------------------------

    usable_3m = (
        rated
        & practice
        & outcome_3
    )

    print()
    print("=== PRACTICE EXPOSURE IN USABLE 3M OBSERVATIONS ===")

    for col in [
        "accepted_rated_count",
        "practice_p50",
        "practice_p75",
        "practice_p90",
        "difficulty_gap_p50",
        "difficulty_gap_p75",
        "difficulty_gap_p90",
    ]:
        print()
        print(col)

        values = df.loc[
            usable_3m,
            col,
        ].dropna()

        if len(values):
            print(values.describe())
        else:
            print("No observations.")

    # ------------------------------------------------------------
    # Upsolve exposure in usable 3m observations
    # ------------------------------------------------------------

    print()
    print("=== UPSOLVE EXPOSURE IN USABLE 3M OBSERVATIONS ===")

    for col in [
        "upsolve_opportunities",
        "strict_upsolves",
        "upsolve_rate",
        "upsolve_rate_24h",
        "upsolve_rate_7d",
        "upsolve_rate_30d",
    ]:
        print()
        print(col)

        values = df.loc[
            usable_3m,
            col,
        ].dropna()

        if len(values):
            print(values.describe())
        else:
            print("No observations.")

    # ------------------------------------------------------------
    # User-level usable observations
    # ------------------------------------------------------------

    print()
    print("=== USER-LEVEL USABLE OBSERVATIONS ===")

    user_counts = (
        df.loc[
            usable_3m
        ]
        .groupby("handle")
        .size()
        .sort_values()
    )

    print(
        "Users with usable 3m practice observations:",
        len(user_counts),
    )

    if len(user_counts):
        print()
        print(user_counts.describe())

        print()
        print("Users with < 3 usable observations:")
        print(
            user_counts[
                user_counts < 3
            ].to_string()
        )

    # ------------------------------------------------------------
    # Save nothing
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()