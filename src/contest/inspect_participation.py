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

    ratings["rating_time"] = pd.to_datetime(
        ratings["rating_update_time_seconds"],
        unit="s",
        utc=True,
    )

    # Only actual contest-participation submissions.
    contestant = submissions[
        submissions["participant_type"] == "CONTESTANT"
    ].copy()

    contest_summary = (
        contestant.groupby("contest_id")
        .agg(
            submission_count=("submission_id", "count"),
            first_submission=("submission_time", "min"),
            last_submission=("submission_time", "max"),
            contest_start_time=("contest_start_time_seconds", "first"),
            participant_id=("participant_id", "first"),
            solved_problems=(
                "verdict",
                lambda x: (x == "OK").sum(),
            ),
        )
        .reset_index()
    )

    contest_summary["contest_start"] = pd.to_datetime(
        contest_summary["contest_start_time"],
        unit="s",
        utc=True,
    )

    # A rated contest should appear in rating history.
    rating_contests = set(ratings["contest_id"])

    contest_summary["in_rating_history"] = (
        contest_summary["contest_id"].isin(rating_contests)
    )

    print("=== CONTEST PARTICIPATION ===")

    print(
        f"Contest IDs with CONTESTANT submissions: "
        f"{len(contest_summary):,}"
    )

    print(
        f"Contest IDs also present in rating history: "
        f"{contest_summary['in_rating_history'].sum():,}"
    )

    print(
        f"Contest IDs NOT present in rating history: "
        f"{(~contest_summary['in_rating_history']).sum():,}"
    )

    print("\n=== PARTICIPATION SUMMARY ===")

    print(
        contest_summary[
            [
                "submission_count",
                "solved_problems",
            ]
        ]
        .describe()
        .to_string()
    )

    print("\n=== CONTESTS NOT IN RATING HISTORY ===")

    not_rated = contest_summary[
        ~contest_summary["in_rating_history"]
    ]

    print(f"Count: {len(not_rated):,}")

    if not_rated.empty:
        print("None.")
    else:
        print(
            not_rated[
                [
                    "contest_id",
                    "contest_start",
                    "submission_count",
                    "solved_problems",
                    "participant_id",
                ]
            ]
            .head(50)
            .to_string(index=False)
        )

    print("\n=== RATED CONTEST SAMPLE ===")

    rated = contest_summary[
        contest_summary["in_rating_history"]
    ].copy()

    print(
        rated[
            [
                "contest_id",
                "contest_start",
                "first_submission",
                "last_submission",
                "submission_count",
                "solved_problems",
                "participant_id",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\n=== CONTEST START TIME CONSISTENCY ===")

    missing_start = contest_summary[
        contest_summary["contest_start_time"].isna()
    ]

    print(
        f"Contest IDs missing startTimeSeconds: "
        f"{len(missing_start):,}"
    )

    print("\n=== PARTICIPANT ID CONSISTENCY ===")

    participant_id_counts = (
        contestant.groupby("contest_id")["participant_id"]
        .nunique(dropna=True)
    )

    multiple_ids = participant_id_counts[
        participant_id_counts > 1
    ]

    print(
        f"Contest IDs with multiple participant IDs: "
        f"{len(multiple_ids):,}"
    )


if __name__ == "__main__":
    main()