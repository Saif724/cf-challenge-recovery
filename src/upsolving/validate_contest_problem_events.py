from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_contest_problem_events.parquet"
)


def main():
    events = pd.read_parquet(EVENT_FILE)

    print("=" * 70)
    print("CONTEST PROBLEM EVENT VALIDATION")
    print("=" * 70)

    errors = []

    # ------------------------------------------------------------
    # 1. Strict upsolve logical definition
    # ------------------------------------------------------------

    expected_strict = (
        events["attempted_during_contest"]
        & ~events["solved_during_contest"]
        & events["solved_after_contest"]
    )

    mismatch = events["strict_upsolve"] != expected_strict

    if mismatch.any():
        errors.append(
            f"strict_upsolve mismatch: {mismatch.sum()} rows"
        )
    else:
        print("PASS: strict_upsolve definition is internally consistent.")

    # ------------------------------------------------------------
    # 2. Strict upsolve must have a delay
    # ------------------------------------------------------------

    bad_delay = (
        events["strict_upsolve"]
        & events["solve_delay_hours"].isna()
    )

    if bad_delay.any():
        errors.append(
            f"strict upsolves with missing delay: {bad_delay.sum()}"
        )
    else:
        print("PASS: every strict upsolve has a solve delay.")

    # ------------------------------------------------------------
    # 3. No negative delays
    # ------------------------------------------------------------

    negative_delay = (
        events["solve_delay_hours"].notna()
        & (events["solve_delay_hours"] < 0)
    )

    if negative_delay.any():
        errors.append(
            f"negative solve delays: {negative_delay.sum()}"
        )
    else:
        print("PASS: no negative solve delays.")

    # ------------------------------------------------------------
    # 4. Solved during implies an OK submission during contest
    # ------------------------------------------------------------

    # This is indirectly checked by the event builder, but
    # we verify the logical combination here.

    impossible = (
        events["solved_during_contest"]
        & ~events["attempted_during_contest"]
    )

    if impossible.any():
        errors.append(
            f"solved_during=True but attempted_during=False: "
            f"{impossible.sum()}"
        )
    else:
        print(
            "PASS: solved_during implies attempted_during."
        )

    # ------------------------------------------------------------
    # 5. Strict upsolve cannot also be solved during
    # ------------------------------------------------------------

    impossible = (
        events["strict_upsolve"]
        & events["solved_during_contest"]
    )

    if impossible.any():
        errors.append(
            f"strict upsolve + solved during: {impossible.sum()}"
        )
    else:
        print(
            "PASS: no strict upsolve is solved during contest."
        )

    # ------------------------------------------------------------
    # 6. Strict upsolve must have attempted during
    # ------------------------------------------------------------

    impossible = (
        events["strict_upsolve"]
        & ~events["attempted_during_contest"]
    )

    if impossible.any():
        errors.append(
            f"strict upsolve without contest attempt: "
            f"{impossible.sum()}"
        )
    else:
        print(
            "PASS: every strict upsolve has a contest attempt."
        )

    # ------------------------------------------------------------
    # 7. Strict upsolve must have solved after
    # ------------------------------------------------------------

    impossible = (
        events["strict_upsolve"]
        & ~events["solved_after_contest"]
    )

    if impossible.any():
        errors.append(
            f"strict upsolve without post-contest solve: "
            f"{impossible.sum()}"
        )
    else:
        print(
            "PASS: every strict upsolve has a post-contest solve."
        )

    # ------------------------------------------------------------
    # 8. Check duplicate event keys
    # ------------------------------------------------------------

    keys = [
        "handle",
        "contest_id",
        "problem_index",
    ]

    duplicates = events.duplicated(keys).sum()

    if duplicates:
        errors.append(
            f"duplicate event keys: {duplicates}"
        )
    else:
        print(
            "PASS: (handle, contest_id, problem_index) is unique."
        )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print()
    print("=" * 70)

    if errors:
        print("VALIDATION FAILED")
        print()

        for error in errors:
            print(f"ERROR: {error}")

        raise SystemExit(1)

    print("ALL VALIDATION CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()