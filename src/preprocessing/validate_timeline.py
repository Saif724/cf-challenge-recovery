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

    print("=== SUBMISSION TIMELINE ===")

    print(
        f"First submission: "
        f"{submissions['submission_time'].min()}"
    )

    print(
        f"Last submission:  "
        f"{submissions['submission_time'].max()}"
    )

    print("\n=== RATING TIMELINE ===")

    print(
        f"First rating update: "
        f"{ratings['rating_time'].min()}"
    )

    print(
        f"Last rating update:  "
        f"{ratings['rating_time'].max()}"
    )

    print("\n=== TIMELINE OVERLAP ===")

    first_submission = submissions["submission_time"].min()
    last_submission = submissions["submission_time"].max()

    first_rating = ratings["rating_time"].min()
    last_rating = ratings["rating_time"].max()

    print(
        f"Submission history starts before rating history: "
        f"{first_submission < first_rating}"
    )

    print(
        f"Submission history ends after rating history: "
        f"{last_submission > last_rating}"
    )

    print("\n=== ACCEPTED SUBMISSIONS AROUND RATING UPDATES ===")

    accepted = submissions[
        submissions["verdict"] == "OK"
    ].copy()

    accepted_rated = accepted[
        accepted["problem_rating"].notna()
    ].copy()

    print(
        f"Rated accepted submissions: "
        f"{len(accepted_rated):,}"
    )

    print("\n=== RATING CHANGE DISTRIBUTION ===")

    ratings["rating_change"] = (
        ratings["new_rating"] - ratings["old_rating"]
    )

    print(
        ratings["rating_change"]
        .describe()
        .to_string()
    )

    print("\n=== RATING CONSISTENCY CHECK ===")

    inconsistent = ratings[
        ratings["rating_change"]
        != ratings["new_rating"] - ratings["old_rating"]
    ]

    print(
        f"Inconsistent rating changes: "
        f"{len(inconsistent)}"
    )

    print("\n=== DUPLICATE CONTESTS IN RATING HISTORY ===")

    duplicate_contests = ratings[
        ratings["contest_id"].duplicated(keep=False)
    ].sort_values("contest_id")

    print(
        f"Rows belonging to duplicated contest IDs: "
        f"{len(duplicate_contests)}"
    )

    if len(duplicate_contests) > 0:
        print(
            duplicate_contests[
                [
                    "contest_id",
                    "contest_name",
                    "rating_time",
                    "old_rating",
                    "new_rating",
                ]
            ]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()