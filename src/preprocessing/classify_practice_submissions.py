from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUBMISSION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_submissions.parquet"
)

RATING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_rating_history.parquet"
)

CONTEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_contests.parquet"
)

EVENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_contest_problem_events.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_submission_activity.parquet"
)


def main():
    submissions = pd.read_parquet(SUBMISSION_FILE)
    ratings = pd.read_parquet(RATING_FILE)
    contests = pd.read_parquet(CONTEST_FILE)
    events = pd.read_parquet(EVENT_FILE)

    print("=" * 70)
    print("CLASSIFYING SUBMISSION ACTIVITY")
    print("=" * 70)

    # ------------------------------------------------------------
    # Normalize IDs
    # ------------------------------------------------------------

    submissions["contest_id"] = pd.to_numeric(
        submissions["contest_id"],
        errors="coerce",
    )

    ratings["contest_id"] = pd.to_numeric(
        ratings["contest_id"],
        errors="coerce",
    )

    contests["contest_id"] = pd.to_numeric(
        contests["contest_id"],
        errors="coerce",
    )

    events["contest_id"] = pd.to_numeric(
        events["contest_id"],
        errors="coerce",
    )

    # ------------------------------------------------------------
    # Contest timing
    # ------------------------------------------------------------

    contest_times = contests[
        [
            "contest_id",
            "start_time",
            "duration_seconds",
        ]
    ].copy()

    contest_times["contest_end"] = (
        contest_times["start_time"]
        + pd.to_timedelta(
            contest_times["duration_seconds"],
            unit="s",
        )
    )

    contest_times = contest_times[
        [
            "contest_id",
            "start_time",
            "contest_end",
        ]
    ]

    # ------------------------------------------------------------
    # Actual participation
    #
    # Rating history contains only contests where the user
    # actually received a rating change.
    # ------------------------------------------------------------

    participation = ratings[
        [
            "handle",
            "contest_id",
        ]
    ].drop_duplicates()

    participation["participated"] = True

    # ------------------------------------------------------------
    # Join contest timing
    # ------------------------------------------------------------

    submissions = submissions.merge(
        contest_times,
        on="contest_id",
        how="left",
    )

    # ------------------------------------------------------------
    # Join actual participation
    # ------------------------------------------------------------

    submissions = submissions.merge(
        participation,
        on=["handle", "contest_id"],
        how="left",
    )

    submissions["participated"] = (
        submissions["participated"]
        .fillna(False)
    )

    # ------------------------------------------------------------
    # Submission timing relative to contest
    # ------------------------------------------------------------

    during = (
        submissions["participated"]
        & submissions["start_time"].notna()
        & (
            submissions["submission_time"]
            >= submissions["start_time"]
        )
        & (
            submissions["submission_time"]
            <= submissions["contest_end"]
        )
    )

    after = (
        submissions["participated"]
        & submissions["contest_end"].notna()
        & (
            submissions["submission_time"]
            > submissions["contest_end"]
        )
    )

    before = (
        submissions["participated"]
        & submissions["start_time"].notna()
        & (
            submissions["submission_time"]
            < submissions["start_time"]
        )
    )

    submissions["during_contest"] = during
    submissions["post_contest"] = after
    submissions["pre_contest"] = before

    # ------------------------------------------------------------
    # Activity classification
    # ------------------------------------------------------------

    submissions["activity_type"] = "practice"

    submissions.loc[
        submissions["during_contest"],
        "activity_type",
    ] = "contest"

    submissions.loc[
        submissions["post_contest"],
        "activity_type",
    ] = "post_contest"

    submissions.loc[
        submissions["pre_contest"],
        "activity_type",
    ] = "pre_contest"

    # ------------------------------------------------------------
    # Accepted submissions
    # ------------------------------------------------------------

    submissions["accepted"] = (
        submissions["verdict"] == "OK"
    )

    # ------------------------------------------------------------
    # Rated problems
    # ------------------------------------------------------------

    submissions["rated_problem"] = (
        submissions["problem_rating"].notna()
    )

    # ------------------------------------------------------------
    # Strict upsolve lookup
    #
    # Only attach this to the exact contest-problem unit.
    # ------------------------------------------------------------

    event_columns = [
        "handle",
        "contest_id",
        "problem_index",
        "strict_upsolve",
        "upsolve_24h",
        "upsolve_7d",
        "upsolve_30d",
    ]

    event_lookup = events[event_columns].copy()

    submissions = submissions.merge(
        event_lookup,
        on=[
            "handle",
            "contest_id",
            "problem_index",
        ],
        how="left",
    )

    for column in [
        "strict_upsolve",
        "upsolve_24h",
        "upsolve_7d",
        "upsolve_30d",
    ]:
        submissions[column] = (
            submissions[column]
            .fillna(False)
        )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    submissions.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    print()
    print("Rows:", len(submissions))
    print("Users:", submissions["handle"].nunique())

    print()
    print("=== ACTIVITY TYPES ===")
    print(
        submissions["activity_type"]
        .value_counts(dropna=False)
    )

    print()
    print("=== ACCEPTED BY ACTIVITY ===")

    accepted_summary = (
        submissions[
            submissions["accepted"]
        ]
        .groupby("activity_type")
        .size()
        .sort_values(ascending=False)
    )

    print(accepted_summary)

    print()
    print("=== RATED ACCEPTED BY ACTIVITY ===")

    rated_accepted_summary = (
        submissions[
            submissions["accepted"]
            & submissions["rated_problem"]
        ]
        .groupby("activity_type")
        .size()
        .sort_values(ascending=False)
    )

    print(rated_accepted_summary)

    print()
    print("=== STRICT UPSOLVE SUBMISSIONS ===")

    print(
        submissions.loc[
            submissions["strict_upsolve"],
            "activity_type",
        ].value_counts()
    )

    print()
    print("=== CONTEST TIMING ===")

    print(
        submissions[
            [
                "during_contest",
                "post_contest",
                "pre_contest",
            ]
        ]
        .sum()
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()