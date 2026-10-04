from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUBMISSIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "submissions.parquet"
)

RATINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ratings.parquet"
)


def main():
    submissions = pd.read_parquet(SUBMISSIONS_FILE)
    ratings = pd.read_parquet(RATINGS_FILE)

    submissions["submission_time"] = pd.to_datetime(
        submissions["creation_time_seconds"],
        unit="s",
        utc=True,
    )

    submissions["contest_start"] = pd.to_datetime(
        submissions["contest_start_time_seconds"],
        unit="s",
        utc=True,
    )

    rated_contests = set(ratings["contest_id"])

    contest_submissions = submissions[
        (submissions["participant_type"] == "CONTESTANT")
        & (submissions["contest_id"].isin(rated_contests))
    ].copy()

    # Duration cannot be inferred safely from the user's submissions.
    # Instead, inspect the time span of submissions relative to contest start.
    contest_submissions["seconds_from_start"] = (
        contest_submissions["creation_time_seconds"]
        - contest_submissions["contest_start_time_seconds"]
    )

    print("=== SUBMISSION OFFSET FROM CONTEST START ===")

    print(
        contest_submissions["seconds_from_start"]
        .describe()
        .to_string()
    )

    print("\n=== NEGATIVE OFFSETS ===")

    before = contest_submissions[
        contest_submissions["seconds_from_start"] < 0
    ]

    print(f"Submissions before contest start: {len(before):,}")

    if not before.empty:
        print(
            before[
                [
                    "contest_id",
                    "problem_index",
                    "creation_time_seconds",
                    "contest_start_time_seconds",
                    "seconds_from_start",
                    "verdict",
                ]
            ]
            .sort_values("seconds_from_start")
            .head(30)
            .to_string(index=False)
        )

    print("\n=== LONGEST SUBMISSION DELAYS ===")

    print(
        contest_submissions[
            [
                "contest_id",
                "problem_index",
                "contest_start",
                "submission_time",
                "seconds_from_start",
                "participant_type",
                "verdict",
            ]
        ]
        .sort_values("seconds_from_start", ascending=False)
        .head(30)
        .to_string(index=False)
    )

    print("\n=== CONTEST-LEVEL MAXIMUM DURATION OBSERVATION ===")

    contest_max = (
        contest_submissions
        .groupby("contest_id")
        .agg(
            contest_start=("contest_start_time_seconds", "first"),
            max_offset=("seconds_from_start", "max"),
            min_offset=("seconds_from_start", "min"),
            submission_count=("submission_id", "count"),
        )
        .reset_index()
        .sort_values("max_offset", ascending=False)
    )

    print(
        contest_max.head(30).to_string(index=False)
    )

    print("\n=== CONTESTS WITH SUBMISSIONS MORE THAN 3 HOURS AFTER START ===")

    long_contests = contest_max[
        contest_max["max_offset"] > 3 * 60 * 60
    ]

    print(
        f"Count: {len(long_contests):,}"
    )

    if not long_contests.empty:
        print(
            long_contests.head(50).to_string(index=False)
        )


if __name__ == "__main__":
    main()