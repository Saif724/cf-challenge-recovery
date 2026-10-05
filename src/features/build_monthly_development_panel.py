from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRACTICE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_monthly_practice_features.parquet"
)

UPSOLVE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_monthly_upsolve_features.parquet"
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
    / "pilot_monthly_development_panel.parquet"
)


def normalize_month(series):
    return (
        pd.to_datetime(series, utc=True)
        .dt.tz_convert(None)
        .dt.to_period("M")
        .dt.to_timestamp()
    )


def normalize_datetime_ns(series):
    return (
        pd.to_datetime(series, utc=True)
        .dt.tz_convert(None)
        .astype("datetime64[ns]")
    )


def main():
    print("=" * 70)
    print("BUILDING MONTHLY DEVELOPMENT PANEL")
    print("=" * 70)

    # ============================================================
    # LOAD DATA
    # ============================================================

    practice = pd.read_parquet(PRACTICE_FILE)
    upsolve = pd.read_parquet(UPSOLVE_FILE)
    ratings = pd.read_parquet(RATING_FILE)

    # ============================================================
    # NORMALIZE DATES
    # ============================================================

    practice["month"] = normalize_month(
        practice["month"]
    )

    upsolve["month"] = normalize_month(
        upsolve["month"]
    )

    ratings["contest_time"] = normalize_datetime_ns(
        ratings["contest_time"]
    )

    ratings["contest_month"] = (
        ratings["contest_time"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    # ============================================================
    # ANALYTICAL COHORT
    #
    # Must have:
    #   1. accepted rated practice
    #   2. rated contest participation
    # ============================================================

    practice_users = set(
        practice["handle"].unique()
    )

    contest_users = set(
        ratings["handle"].unique()
    )

    cohort = sorted(
        practice_users & contest_users
    )

    print()
    print("Practice users:", len(practice_users))
    print("Contest users:", len(contest_users))
    print("Analytical cohort:", len(cohort))

    if len(cohort) != 177:
        raise ValueError(
            f"Unexpected analytical cohort size: {len(cohort)}"
        )

    # ============================================================
    # RESTRICT DATASETS TO COHORT
    # ============================================================

    practice = practice[
        practice["handle"].isin(cohort)
    ].copy()

    upsolve = upsolve[
        upsolve["handle"].isin(cohort)
    ].copy()

    ratings = ratings[
        ratings["handle"].isin(cohort)
    ].copy()

    # ============================================================
    # PRIMARY PRACTICE DEFINITION
    #
    # CLEAN is primary.
    # BROAD remains available for robustness later.
    # ============================================================

    practice_clean = practice[
        practice["practice_definition"] == "clean"
    ].copy()

    practice_clean = (
        practice_clean[
            [
                "handle",
                "month",
                "accepted_rated_count",
                "practice_p50",
                "practice_p75",
                "practice_p90",
                "difficulty_gap_p50",
                "difficulty_gap_p75",
                "difficulty_gap_p90",
                "rating_before_month",
            ]
        ]
        .drop_duplicates(
            subset=["handle", "month"]
        )
    )

    # ============================================================
    # UPSOLVE MONTHLY FEATURES
    # ============================================================

    upsolve = (
        upsolve[
            [
                "handle",
                "month",
                "upsolve_opportunities",
                "strict_upsolves",
                "upsolves_24h",
                "upsolves_7d",
                "upsolves_30d",
                "contests_with_opportunity",
                "upsolve_rate",
                "upsolve_rate_24h",
                "upsolve_rate_7d",
                "upsolve_rate_30d",
            ]
        ]
        .drop_duplicates(
            subset=["handle", "month"]
        )
    )

    # ============================================================
    # CONTEST ACTIVITY
    #
    # Authoritative source = rating history.
    # ============================================================

    contest_monthly = (
        ratings
        .groupby(
            ["handle", "contest_month"],
            as_index=False,
        )
        .agg(
            contest_count_month=(
                "contest_id",
                "nunique",
            ),
            rating_last_contest=(
                "new_rating",
                "last",
            ),
            rating_first_contest=(
                "new_rating",
                "first",
            ),
        )
        .rename(
            columns={
                "contest_month": "month",
            }
        )
    )

    # ============================================================
    # GLOBAL OBSERVATION WINDOW
    # ============================================================

    practice_min = practice["month"].min()
    practice_max = practice["month"].max()

    upsolve_min = upsolve["month"].min()
    upsolve_max = upsolve["month"].max()

    contest_min = ratings["contest_month"].min()
    contest_max = ratings["contest_month"].max()

    observation_start = min(
        practice_min,
        upsolve_min,
        contest_min,
    )

    observation_end = max(
        practice_max,
        upsolve_max,
        contest_max,
    )

    print()
    print("Observation start:", observation_start)
    print("Observation end:", observation_end)

    # ============================================================
    # COMPLETE USER-MONTH GRID
    # ============================================================

    months = pd.date_range(
        start=observation_start,
        end=observation_end,
        freq="MS",
    )

    calendar = pd.MultiIndex.from_product(
        [
            cohort,
            months,
        ],
        names=["handle", "month"],
    ).to_frame(index=False)

    print()
    print("Users:", len(cohort))
    print("Months:", len(months))
    print("Potential user-month rows:", len(calendar))

    # ============================================================
    # BUILD MAIN BEHAVIORAL PANEL
    # ============================================================

    panel = calendar.merge(
        practice_clean,
        on=["handle", "month"],
        how="left",
    )

    panel = panel.merge(
        upsolve,
        on=["handle", "month"],
        how="left",
    )

    panel = panel.merge(
        contest_monthly,
        on=["handle", "month"],
        how="left",
    )

    # ============================================================
    # CONTEST COUNTS
    # ============================================================

    panel["contest_count_month"] = (
        panel["contest_count_month"]
        .fillna(0)
        .astype(int)
    )

    panel["contests_participated"] = (
        panel["contest_count_month"]
    )

    # ============================================================
    # ZERO-FILL BEHAVIORAL COUNTS
    # ============================================================

    panel["accepted_rated_count"] = (
        panel["accepted_rated_count"]
        .fillna(0)
        .astype(int)
    )

    for column in [
        "upsolve_opportunities",
        "strict_upsolves",
        "upsolves_24h",
        "upsolves_7d",
        "upsolves_30d",
        "contests_with_opportunity",
    ]:
        panel[column] = (
            panel[column]
            .fillna(0)
            .astype(int)
        )

    for column in [
        "upsolve_rate",
        "upsolve_rate_24h",
        "upsolve_rate_7d",
        "upsolve_rate_30d",
    ]:
        panel[column] = (
            panel[column]
            .fillna(0.0)
        )

    # ============================================================
    # RATING STATE AT MONTH END
    #
    # IMPORTANT:
    # Build this separately and MERGE it into panel.
    # Do NOT replace panel with merge_asof output.
    # ============================================================

    rating_events = ratings[
        [
            "handle",
            "contest_time",
            "new_rating",
        ]
    ].copy()

    rating_events["contest_time"] = (
        normalize_datetime_ns(
            rating_events["contest_time"]
        )
    )

    rating_events = rating_events.sort_values(
        ["contest_time", "handle"]
    ).reset_index(drop=True)

    month_ends = (
        calendar[
            ["handle", "month"]
        ]
        .drop_duplicates()
        .copy()
    )

    month_ends["month_end"] = (
        month_ends["month"]
        + pd.offsets.MonthEnd(0)
        + pd.Timedelta(days=1)
        - pd.Timedelta(microseconds=1)
    )

    month_ends["month_end"] = (
        normalize_datetime_ns(
            month_ends["month_end"]
        )
    )

    month_ends = month_ends.sort_values(
        ["month_end", "handle"]
    ).reset_index(drop=True)

    rating_state = pd.merge_asof(
        month_ends,
        rating_events,
        left_on="month_end",
        right_on="contest_time",
        by="handle",
        direction="backward",
    )

    rating_state = rating_state.rename(
        columns={
            "new_rating": "rating_at_month_end",
        }
    )

    rating_state = rating_state[
        [
            "handle",
            "month",
            "rating_at_month_end",
        ]
    ]

    # Merge rating state INTO the existing behavioral panel.
    panel = panel.merge(
        rating_state,
        on=["handle", "month"],
        how="left",
    )

    # ============================================================
    # FORWARD RATING OUTCOMES
    #
    # For each month t:
    #
    # baseline:
    #   latest rating available by month-end
    #
    # outcome:
    #   first rated contest strictly after month-end and within
    #   the specified calendar horizon.
    #
    # If there is no contest in the horizon -> NaN.
    #
    # If there is no baseline rating -> NaN.
    # ============================================================

    outcome_rows = []

    for horizon_months in [1, 3, 6]:

        horizon_rows = calendar.copy()

        horizon_rows["window_end"] = (
            horizon_rows["month"]
            + pd.DateOffset(
                months=horizon_months + 1
            )
            - pd.Timedelta(microseconds=1)
        )

        horizon_rows["window_end"] = (
            normalize_datetime_ns(
                horizon_rows["window_end"]
            )
        )

        left = horizon_rows[
            [
                "handle",
                "month",
                "window_end",
            ]
        ].copy()

        # First instant of next month.
        left["search_time"] = (
            left["month"]
            + pd.offsets.MonthEnd(0)
            + pd.Timedelta(days=1)
        )

        left["search_time"] = (
            normalize_datetime_ns(
                left["search_time"]
            )
        )

        left = left.sort_values(
            ["search_time", "handle"]
        ).reset_index(drop=True)

        right = ratings[
            [
                "handle",
                "contest_time",
                "new_rating",
            ]
        ].copy()

        right["contest_time"] = (
            normalize_datetime_ns(
                right["contest_time"]
            )
        )

        right = right.sort_values(
            ["contest_time", "handle"]
        ).reset_index(drop=True)

        first_future = pd.merge_asof(
            left,
            right,
            left_on="search_time",
            right_on="contest_time",
            by="handle",
            direction="forward",
            allow_exact_matches=False,
        )

        valid_future = (
            first_future["contest_time"].notna()
            & (
                first_future["contest_time"]
                <= first_future["window_end"]
            )
        )

        outcome_name = (
            f"rating_gain_{horizon_months}m"
        )

        first_future[outcome_name] = np.nan

        # --------------------------------------------------------
        # Add baseline rating.
        # --------------------------------------------------------

        baseline = panel[
            [
                "handle",
                "month",
                "rating_at_month_end",
            ]
        ].copy()

        first_future = first_future.merge(
            baseline,
            on=["handle", "month"],
            how="left",
        )

        valid_with_baseline = (
            valid_future
            & first_future[
                "rating_at_month_end"
            ].notna()
        )

        first_future.loc[
            valid_with_baseline,
            outcome_name,
        ] = (
            first_future.loc[
                valid_with_baseline,
                "new_rating",
            ]
            - first_future.loc[
                valid_with_baseline,
                "rating_at_month_end",
            ]
        )

        outcome_rows.append(
            first_future[
                [
                    "handle",
                    "month",
                    outcome_name,
                ]
            ]
        )

    # ============================================================
    # MERGE OUTCOMES
    # ============================================================

    for outcome in outcome_rows:
        panel = panel.merge(
            outcome,
            on=["handle", "month"],
            how="left",
        )

    # ============================================================
    # INACTIVITY INDICATOR
    #
    # This is only an indicator.
    # It does NOT turn missing future contests into zero rating
    # gains.
    # ============================================================

    panel["active_contest_month"] = (
        panel["contest_count_month"] > 0
    )

    # ============================================================
    # FINAL SORT
    # ============================================================

    panel = panel.sort_values(
        ["handle", "month"]
    ).reset_index(drop=True)

    # ============================================================
    # DIAGNOSTICS
    # ============================================================

    print()
    print("=== PANEL ===")
    print("Rows:", len(panel))
    print("Users:", panel["handle"].nunique())
    print("Months:", panel["month"].nunique())

    print()
    print("=== RATING COVERAGE ===")

    rating_nonnull = (
        panel["rating_at_month_end"]
        .notna()
        .sum()
    )

    print(
        "Rows with rating:",
        rating_nonnull,
    )

    print(
        "Rows without rating:",
        len(panel) - rating_nonnull,
    )

    print(
        "Rating coverage:",
        f"{100 * rating_nonnull / len(panel):.2f}%",
    )

    print()
    print("=== OUTCOME COVERAGE ===")

    for column in [
        "rating_gain_1m",
        "rating_gain_3m",
        "rating_gain_6m",
    ]:
        nonnull = panel[column].notna().sum()

        print()
        print(column)

        if nonnull > 0:
            print(
                panel[column].describe()
            )
        else:
            print("No non-null observations.")

        print(
            "Non-null:",
            nonnull,
        )

        print(
            "Coverage:",
            f"{100 * nonnull / len(panel):.2f}%",
        )

    print()
    print("=== BEHAVIORAL COVERAGE ===")

    practice_months = (
        panel["accepted_rated_count"] > 0
    ).sum()

    upsolve_months = (
        panel["upsolve_opportunities"] > 0
    ).sum()

    contest_months = (
        panel["active_contest_month"]
    ).sum()

    print(
        "Months with practice:",
        practice_months,
    )

    print(
        "Months with upsolve opportunity:",
        upsolve_months,
    )

    print(
        "Months with contest:",
        contest_months,
    )

    # ============================================================
    # ADDITIONAL USER-LEVEL DIAGNOSTICS
    # ============================================================

    print()
    print("=== USER-LEVEL COVERAGE ===")

    users_with_rating = (
        panel.loc[
            panel["rating_at_month_end"].notna(),
            "handle",
        ]
        .nunique()
    )

    users_with_3m_outcome = (
        panel.loc[
            panel["rating_gain_3m"].notna(),
            "handle",
        ]
        .nunique()
    )

    users_with_6m_outcome = (
        panel.loc[
            panel["rating_gain_6m"].notna(),
            "handle",
        ]
        .nunique()
    )

    print(
        "Users with at least one rated month:",
        users_with_rating,
        "/",
        len(cohort),
    )

    print(
        "Users with at least one 3m outcome:",
        users_with_3m_outcome,
        "/",
        len(cohort),
    )

    print(
        "Users with at least one 6m outcome:",
        users_with_6m_outcome,
        "/",
        len(cohort),
    )

    # ============================================================
    # SAVE
    # ============================================================

    panel.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()