from pathlib import Path
import json

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

CONTESTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "contests.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "contest_events.parquet"
)


def main():
    submissions = pd.read_parquet(SUBMISSIONS_FILE)
    ratings = pd.read_parquet(RATINGS_FILE)

    with open(CONTESTS_FILE, encoding="utf-8") as f:
        contests = pd.DataFrame(json.load(f))

    submissions["submission_time"] = pd.to_datetime(
        submissions["creation_time_seconds"],
        unit="s",
        utc=True,
    )

    # Only contests that produced an official rating update.
    rated_contests = set(ratings["contest_id"])

    contests = contests[
        contests["id"].isin(rated_contests)
    ].copy()

    contests["contest_end_time_seconds"] = (
        contests["startTimeSeconds"]
        + contests["durationSeconds"]
    )

    # We only use submissions associated with rated contests.
    submissions = submissions[
        submissions["contest_id"].isin(rated_contests)
    ].copy()

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

    # IMPORTANT:
    # Contest-time activity is defined by the official contest window,
    # not merely by participant_type.
    submissions["during_contest"] = (
        (submissions["creation_time_seconds"]
         >= submissions["startTimeSeconds"])
        &
        (submissions["creation_time_seconds"]
         < submissions["contest_end_time_seconds"])
    )

    submissions["after_contest"] = (
        submissions["creation_time_seconds"]
        >= submissions["contest_end_time_seconds"]
    )

    # We are reconstructing actual participation.
    # CONTESTANT submissions during the official contest window
    # are contest attempts.
    contest_submissions = submissions[
        (submissions["participant_type"] == "CONTESTANT")
        & (submissions["during_contest"])
    ].copy()

    # Problems attempted during the actual contest.
    attempted = (
        contest_submissions
        .groupby(
            [
                "handle",
                "contest_id",
                "problem_index",
            ],
            as_index=False,
        )
        .agg(
            attempted_during_contest=(
                "submission_id",
                "count",
            ),
            solved_during_contest=(
                "verdict",
                lambda x: (x == "OK").any(),
            ),
            first_contest_submission=(
                "submission_time",
                "min",
            ),
            last_contest_submission=(
                "submission_time",
                "max",
            ),
            problem_name=(
                "problem_name",
                "first",
            ),
            problem_rating=(
                "problem_rating",
                "first",
            ),
        )
    )

    # All later accepted submissions for the same
    # user + contest + problem.
    after = submissions[
        (submissions["after_contest"])
        & (submissions["verdict"] == "OK")
    ].copy()

    later_solved = (
        after
        .groupby(
            [
                "handle",
                "contest_id",
                "problem_index",
            ],
            as_index=False,
        )
        .agg(
            first_solve_after_contest=(
                "submission_time",
                "min",
            ),
        )
    )

    events = attempted.merge(
        later_solved,
        on=[
            "handle",
            "contest_id",
            "problem_index",
        ],
        how="left",
    )

    events["solved_after_contest"] = (
        events["first_solve_after_contest"].notna()
    )

    events["attempted_but_unsolved"] = (
        events["attempted_during_contest"] > 0
    ) & (
        ~events["solved_during_contest"]
    )

    events["strict_upsolve"] = (
        events["attempted_but_unsolved"]
        & events["solved_after_contest"]
    )

    events["solve_delay_hours"] = (
        (
            events["first_solve_after_contest"]
            - events["last_contest_submission"]
        )
        .dt.total_seconds()
        / 3600
    )

    # Keep only useful columns.
    events = events[
        [
            "handle",
            "contest_id",
            "problem_index",
            "problem_name",
            "problem_rating",
            "attempted_during_contest",
            "solved_during_contest",
            "attempted_but_unsolved",
            "solved_after_contest",
            "strict_upsolve",
            "first_contest_submission",
            "last_contest_submission",
            "first_solve_after_contest",
            "solve_delay_hours",
        ]
    ]

    events = events.sort_values(
        [
            "handle",
            "contest_id",
            "problem_index",
        ]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    events.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("=== CONTEST EVENTS ===")
    print(f"Rows: {len(events):,}")
    print(f"Contests: {events['contest_id'].nunique():,}")
    print(f"Problems: {events['problem_index'].nunique():,}")

    print("\n=== STATES ===")

    print(
        events[
            [
                "solved_during_contest",
                "attempted_but_unsolved",
                "solved_after_contest",
                "strict_upsolve",
            ]
        ]
        .value_counts()
        .to_string()
    )

    print("\n=== SUMMARY ===")

    print(
        f"Attempted during contest: "
        f"{events['attempted_during_contest'].gt(0).sum():,}"
    )

    print(
        f"Solved during contest: "
        f"{events['solved_during_contest'].sum():,}"
    )

    print(
        f"Attempted but unsolved: "
        f"{events['attempted_but_unsolved'].sum():,}"
    )

    print(
        f"Solved after contest: "
        f"{events['solved_after_contest'].sum():,}"
    )

    print(
        f"Strict upsolves: "
        f"{events['strict_upsolve'].sum():,}"
    )

    print("\n=== STRICT UPSOLVE RATE ===")

    denominator = events["attempted_but_unsolved"].sum()

    if denominator:
        rate = (
            events["strict_upsolve"].sum()
            / denominator
        )

        print(
            f"{rate:.2%}"
            f" ({events['strict_upsolve'].sum():,}"
            f" / {denominator:,})"
        )
    else:
        print("No attempted-but-unsolved problems.")

    print("\n=== EXAMPLES OF STRICT UPSOLVES ===")

    examples = events[
        events["strict_upsolve"]
    ].head(30)

    if examples.empty:
        print("None found.")
    else:
        print(
            examples[
                [
                    "contest_id",
                    "problem_index",
                    "problem_name",
                    "problem_rating",
                    "attempted_during_contest",
                    "last_contest_submission",
                    "first_solve_after_contest",
                    "solve_delay_hours",
                ]
            ].to_string(index=False)
        )

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()