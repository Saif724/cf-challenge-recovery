from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_contest_problem_events.parquet"
)

SUBMISSION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_submissions.parquet"
)


def show_event(event, submissions):
    handle = event["handle"]
    contest_id = event["contest_id"]
    problem_index = event["problem_index"]

    print()
    print("-" * 70)
    print(
        f"{handle} | contest={contest_id} | "
        f"problem={problem_index} | "
        f"{event['problem_name']}"
    )

    print(
        f"Attempted during: {event['attempted_during_contest']}"
    )
    print(
        f"Solved during:    {event['solved_during_contest']}"
    )
    print(
        f"Solved after:     {event['solved_after_contest']}"
    )
    print(
        f"Strict upsolve:   {event['strict_upsolve']}"
    )
    print(
        f"Delay hours:      {event['solve_delay_hours']}"
    )

    group = submissions[
        (submissions["handle"] == handle)
        & (submissions["contest_id"] == contest_id)
        & (submissions["problem_index"] == problem_index)
    ].copy()

    group = group.sort_values("submission_time")

    print()
    print("SUBMISSIONS:")

    for _, row in group.iterrows():
        print(
            f"  {row['submission_time']} | "
            f"{row['verdict']}"
        )


def main():
    events = pd.read_parquet(EVENT_FILE)
    submissions = pd.read_parquet(SUBMISSION_FILE)

    print("=" * 70)
    print("CONTEST PROBLEM EVENT INSPECTION")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Strict upsolve examples
    # ------------------------------------------------------------

    upsolves = events[
        events["strict_upsolve"]
    ]

    print()
    print("=== STRICT UPSOLVE EXAMPLES ===")

    for _, event in upsolves.head(5).iterrows():
        show_event(event, submissions)

    # ------------------------------------------------------------
    # 2. Solved during contest
    # ------------------------------------------------------------

    solved_during = events[
        events["solved_during_contest"]
    ]

    print()
    print("=== SOLVED DURING CONTEST EXAMPLES ===")

    for _, event in solved_during.head(3).iterrows():
        show_event(event, submissions)

    # ------------------------------------------------------------
    # 3. Solved after without contest attempt
    # ------------------------------------------------------------

    post_only = events[
        (~events["attempted_during_contest"])
        & events["solved_after_contest"]
    ]

    print()
    print("=== POST-CONTEST SOLVE WITHOUT CONTEST ATTEMPT ===")

    for _, event in post_only.head(3).iterrows():
        show_event(event, submissions)

    # ------------------------------------------------------------
    # 4. Attempted but never solved
    # ------------------------------------------------------------

    never_solved = events[
        events["attempted_during_contest"]
        & ~events["solved_during_contest"]
        & ~events["solved_after_contest"]
    ]

    print()
    print("=== ATTEMPTED BUT NEVER SOLVED ===")

    for _, event in never_solved.head(3).iterrows():
        show_event(event, submissions)

    # ------------------------------------------------------------
    # 5. Delay distribution
    # ------------------------------------------------------------

    delays = events.loc[
        events["strict_upsolve"],
        "solve_delay_hours",
    ].dropna()

    print()
    print("=== DELAY QUANTILES ===")
    print(
        delays.quantile(
            [
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )


if __name__ == "__main__":
    main()