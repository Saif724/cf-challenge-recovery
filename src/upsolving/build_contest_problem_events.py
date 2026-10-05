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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "pilot_contest_problem_events.parquet"
)


def main():
    print("=" * 70)
    print("BUILDING CONTEST PROBLEM EVENTS")
    print("=" * 70)

    submissions = pd.read_parquet(SUBMISSION_FILE)
    ratings = pd.read_parquet(RATING_FILE)
    contests = pd.read_parquet(CONTEST_FILE)

    # ------------------------------------------------------------
    # Only consider contests actually participated in by pilot users.
    # ------------------------------------------------------------

    participation = (
        ratings[
            [
                "handle",
                "contest_id",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    print()
    print("=== INPUT ===")
    print(f"Submissions: {len(submissions):,}")
    print(f"Participation rows: {len(participation):,}")
    print(f"Contest metadata rows: {len(contests):,}")

    # ------------------------------------------------------------
    # Attach contest start/end times to user participation.
    # ------------------------------------------------------------

    participation = participation.merge(
        contests[
            [
                "contest_id",
                "start_time",
                "end_time",
            ]
        ],
        on="contest_id",
        how="left",
        validate="many_to_one",
    )

    print()
    print("Missing contest metadata after join:")
    print(
        participation[
            ["start_time", "end_time"]
        ]
        .isna()
        .any(axis=1)
        .sum()
    )

    # ------------------------------------------------------------
    # Attach contest timing to submissions.
    #
    # IMPORTANT:
    # Only submissions from contests the user actually
    # participated in are considered here.
    # ------------------------------------------------------------

    contest_submissions = submissions[
        submissions["contest_id"].notna()
    ].copy()

    contest_submissions = contest_submissions.merge(
        participation,
        on=["handle", "contest_id"],
        how="inner",
        validate="many_to_many",
    )

    print()
    print(
        "Submissions belonging to contests actually "
        "participated in:",
        f"{len(contest_submissions):,}",
    )

    # ------------------------------------------------------------
    # Classify submission timing.
    # ------------------------------------------------------------

    during = (
        (contest_submissions["submission_time"]
         >= contest_submissions["start_time"])
        &
        (contest_submissions["submission_time"]
         <= contest_submissions["end_time"])
    )

    after = (
        contest_submissions["submission_time"]
        > contest_submissions["end_time"]
    )

    contest_submissions["during_contest"] = during
    contest_submissions["after_contest"] = after

    print()
    print("=== SUBMISSION TIMING ===")
    print(
        "During contest:",
        int(during.sum()),
    )
    print(
        "After contest:",
        int(after.sum()),
    )
    print(
        "Before contest:",
        int(
            (
                contest_submissions["submission_time"]
                < contest_submissions["start_time"]
            ).sum()
        ),
    )

    # ------------------------------------------------------------
    # Restrict to submissions during or after the contest.
    # ------------------------------------------------------------

    relevant = contest_submissions[
        contest_submissions["during_contest"]
        | contest_submissions["after_contest"]
    ].copy()

    # Only submissions to identifiable problems.
    relevant = relevant[
        relevant["problem_index"].notna()
    ].copy()

    # ------------------------------------------------------------
    # Build one row per user + contest + problem.
    # ------------------------------------------------------------

    group_columns = [
        "handle",
        "contest_id",
        "problem_index",
    ]

    rows = []

    for keys, group in relevant.groupby(
        group_columns,
        sort=False,
    ):
        handle, contest_id, problem_index = keys

        during_group = group[
            group["during_contest"]
        ]

        after_group = group[
            group["after_contest"]
        ]

        attempted_during = not during_group.empty

        solved_during = (
            (during_group["verdict"] == "OK").any()
        )

        solved_after = (
            (after_group["verdict"] == "OK").any()
        )

        first_contest_submission = (
            during_group["submission_time"].min()
            if not during_group.empty
            else pd.NaT
        )

        last_contest_submission = (
            during_group["submission_time"].max()
            if not during_group.empty
            else pd.NaT
        )

        first_solve_after = pd.NaT

        if solved_after:
            first_solve_after = after_group.loc[
                after_group["verdict"] == "OK",
                "submission_time",
            ].min()

        solve_delay_hours = None

        if (
            attempted_during
            and not solved_during
            and pd.notna(first_solve_after)
        ):
            contest_end = group["end_time"].iloc[0]

            solve_delay_hours = (
                first_solve_after - contest_end
            ).total_seconds() / 3600.0

        strict_upsolve = (
            attempted_during
            and not solved_during
            and solved_after
        )

        problem_name = group["problem_name"].dropna()

        if not problem_name.empty:
            problem_name = problem_name.iloc[0]
        else:
            problem_name = None

        problem_rating = group["problem_rating"].dropna()

        if not problem_rating.empty:
            problem_rating = problem_rating.iloc[0]
        else:
            problem_rating = None

        rows.append(
            {
                "handle": handle,
                "contest_id": contest_id,
                "problem_index": problem_index,
                "problem_name": problem_name,
                "problem_rating": problem_rating,

                "contest_start": group["start_time"].iloc[0],
                "contest_end": group["end_time"].iloc[0],

                "attempted_during_contest": attempted_during,
                "solved_during_contest": solved_during,

                "attempt_count_during_contest": len(
                    during_group
                ),

                "first_contest_submission":
                    first_contest_submission,

                "last_contest_submission":
                    last_contest_submission,

                "solved_after_contest": solved_after,

                "first_solve_after_contest":
                    first_solve_after,

                "solve_delay_hours":
                    solve_delay_hours,

                "strict_upsolve":
                    strict_upsolve,
            }
        )

    events = pd.DataFrame(rows)

    if not events.empty:
        events = events.sort_values(
            [
                "handle",
                "contest_start",
                "contest_id",
                "problem_index",
            ]
        ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    events.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("CONTEST PROBLEM EVENT BUILD COMPLETE")
    print("=" * 70)

    print(f"Rows: {len(events):,}")
    print(
        f"Users: {events['handle'].nunique():,}"
    )
    print(
        f"Contests: {events['contest_id'].nunique():,}"
    )

    print()
    print("=== EVENT COUNTS ===")

    print(
        "Attempted during contest:",
        int(
            events[
                "attempted_during_contest"
            ].sum()
        ),
    )

    print(
        "Solved during contest:",
        int(
            events[
                "solved_during_contest"
            ].sum()
        ),
    )

    print(
        "Attempted but unsolved:",
        int(
            (
                events["attempted_during_contest"]
                &
                ~events["solved_during_contest"]
            ).sum()
        ),
    )

    print(
        "Solved after contest:",
        int(
            events[
                "solved_after_contest"
            ].sum()
        ),
    )

    print(
        "Strict upsolves:",
        int(
            events[
                "strict_upsolve"
            ].sum()
        ),
    )

    print()
    print("=== UPSOLVE DELAY ===")

    delays = events.loc[
        events["strict_upsolve"],
        "solve_delay_hours",
    ].dropna()

    print(delays.describe())

    print()
    print("Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()