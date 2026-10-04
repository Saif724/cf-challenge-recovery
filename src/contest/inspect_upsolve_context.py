from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUBMISSIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "submissions.parquet"
)

EVENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "contest_events.parquet"
)

CONTESTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "contests.json"
)


def main():
    submissions = pd.read_parquet(SUBMISSIONS_FILE)
    events = pd.read_parquet(EVENTS_FILE)

    contests = pd.read_json(CONTESTS_FILE)

    submissions["submission_time"] = pd.to_datetime(
        submissions["creation_time_seconds"],
        unit="s",
        utc=True,
    )

    contests["contest_end_time_seconds"] = (
        contests["startTimeSeconds"]
        + contests["durationSeconds"]
    )

    submissions = submissions.merge(
        contests[
            [
                "id",
                "startTimeSeconds",
                "durationSeconds",
                "contest_end_time_seconds",
            ]
        ],
        left_on="contest_id",
        right_on="id",
        how="inner",
    )

    # Only accepted submissions after the official contest ended.
    after = submissions[
        (submissions["creation_time_seconds"]
         >= submissions["contest_end_time_seconds"])
        & (submissions["verdict"] == "OK")
    ].copy()

    after = after.merge(
        events[
            [
                "handle",
                "contest_id",
                "problem_index",
                "problem_name",
                "problem_rating",
                "strict_upsolve",
            ]
        ],
        on=[
            "handle",
            "contest_id",
            "problem_index",
        ],
        how="inner",
    )

    print("=== LATER ACCEPTED SUBMISSIONS ===")

    print(
        after[
            "participant_type"
        ]
        .value_counts()
        .to_string()
    )

    print("\n=== LATER ACCEPTED SUBMISSIONS FOR STRICT UPSOLVES ===")

    strict = after[
        after["strict_upsolve"]
    ].copy()

    print(
        strict[
            "participant_type"
        ]
        .value_counts()
        .to_string()
    )

    print("\n=== STRICT UPSOLVE CONTEXT ===")

    print(
        strict[
            [
                "contest_id",
                "problem_index",
                "problem_name",
                "problem_rating",
                "participant_type",
                "contest_id",
                "creation_time_seconds",
            ]
        ]
        .sort_values(
            [
                "contest_id",
                "problem_index",
            ]
        )
        .to_string(index=False)
    )

    print("\n=== PARTICIPANT TYPE × STRICT UPSOLVE ===")

    print(
        pd.crosstab(
            after["participant_type"],
            after["strict_upsolve"],
        )
        .to_string()
    )


if __name__ == "__main__":
    main()