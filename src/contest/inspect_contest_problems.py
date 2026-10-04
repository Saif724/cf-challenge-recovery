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

    ratings["rating_time"] = pd.to_datetime(
        ratings["rating_update_time_seconds"],
        unit="s",
        utc=True,
    )

    # Only contests that produced an official rating update.
    rated_contests = set(ratings["contest_id"])

    contest = submissions[
        (submissions["participant_type"] == "CONTESTANT")
        & (submissions["contest_id"].isin(rated_contests))
    ].copy()

    # We need to inspect problems at the problem level.
    problems = (
        contest.groupby(
            [
                "contest_id",
                "problem_index",
                "problem_name",
                "problem_rating",
            ],
            dropna=False,
        )
        .agg(
            submission_count=("submission_id", "count"),
            first_submission=("submission_time", "min"),
            last_submission=("submission_time", "max"),
            accepted_count=(
                "verdict",
                lambda x: (x == "OK").sum(),
            ),
            participant_types=(
                "participant_type",
                lambda x: ",".join(sorted(set(x))),
            ),
        )
        .reset_index()
    )

    # A problem is considered solved during the contest
    # if there was an accepted submission while the user
    # was participating as CONTESTANT.
    problems["solved_during_contest"] = (
        problems["accepted_count"] > 0
    )

    problems["attempted_during_contest"] = True

    print("=== RATED CONTEST PROBLEM EVENTS ===")
    print(f"Rated contests with CONTESTANT submissions: "
          f"{contest['contest_id'].nunique():,}")
    print(f"Contest problems submitted to: {len(problems):,}")

    print("\n=== PROBLEM-LEVEL STATES ===")

    print(
        problems[
            [
                "solved_during_contest",
                "attempted_during_contest",
            ]
        ]
        .value_counts()
        .to_string()
    )

    print("\n=== ATTEMPTED BUT UNSOLVED ===")

    attempted_unsolved = problems[
        (problems["attempted_during_contest"])
        & (~problems["solved_during_contest"])
    ]

    print(
        f"Attempted but unsolved contest problems: "
        f"{len(attempted_unsolved):,}"
    )

    print("\n=== CONTEST EXAMPLES ===")

    # Pick contests with a reasonable number of submissions
    # and at least one attempted-but-unsolved problem.
    contest_counts = (
        attempted_unsolved.groupby("contest_id")
        .size()
        .sort_values(ascending=False)
    )

    selected_contests = contest_counts.head(10).index.tolist()

    for contest_id in selected_contests:
        print("\n" + "=" * 80)
        print(f"CONTEST {contest_id}")

        contest_problems = problems[
            problems["contest_id"] == contest_id
        ].sort_values("problem_index")

        print(
            contest_problems[
                [
                    "problem_index",
                    "problem_name",
                    "problem_rating",
                    "submission_count",
                    "accepted_count",
                    "first_submission",
                    "last_submission",
                    "solved_during_contest",
                ]
            ].to_string(index=False)
        )

    print("\n=== CONTEST WITH NO ACCEPTED SOLUTION ===")

    no_accept = (
        problems.groupby("contest_id")["solved_during_contest"]
        .any()
    )

    no_accept_contests = no_accept[~no_accept]

    print(
        f"Rated contests where no submitted problem was solved: "
        f"{len(no_accept_contests):,}"
    )

    print("\n=== PROBLEM RATING COVERAGE ===")

    print(
        problems["problem_rating"]
        .notna()
        .value_counts()
        .rename(
            {
                True: "rated",
                False: "unrated",
            }
        )
        .to_string()
    )


if __name__ == "__main__":
    main()