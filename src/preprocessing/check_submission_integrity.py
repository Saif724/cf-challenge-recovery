from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_submissions.parquet"
)


def main():
    df = pd.read_parquet(INPUT_FILE)

    print("=" * 70)
    print("SUBMISSION INTEGRITY CHECK")
    print("=" * 70)

    print()
    print("=== BASIC ===")
    print(f"Rows: {len(df):,}")
    print(f"Users: {df['handle'].nunique():,}")

    # ------------------------------------------------------------
    # Submission ID uniqueness
    # ------------------------------------------------------------

    print()
    print("=== SUBMISSION ID UNIQUENESS ===")

    missing_ids = df["submission_id"].isna().sum()
    duplicate_ids = df["submission_id"].duplicated().sum()

    print(f"Missing submission IDs: {missing_ids:,}")
    print(f"Duplicate submission IDs: {duplicate_ids:,}")
    print(
        "Unique submission IDs:",
        df["submission_id"].nunique(),
    )

    # ------------------------------------------------------------
    # Contest/problem multiplicity
    # ------------------------------------------------------------

    print()
    print("=== USER + CONTEST + PROBLEM ===")

    contest_problem = (
        df[
            df["contest_id"].notna()
            & df["problem_index"].notna()
        ]
        .groupby(
            ["handle", "contest_id", "problem_index"],
            dropna=False,
        )
        .size()
        .rename("submission_count")
    )

    print(
        f"Unique user-contest-problem combinations: "
        f"{len(contest_problem):,}"
    )

    print(
        f"Combinations with >1 submission: "
        f"{(contest_problem > 1).sum():,}"
    )

    print(
        f"Combinations with exactly 1 submission: "
        f"{(contest_problem == 1).sum():,}"
    )

    print()
    print("Submission-count distribution per user-contest-problem:")

    print(
        contest_problem.describe(
            percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
        )
    )

    # ------------------------------------------------------------
    # Accepted duplicate behavior
    # ------------------------------------------------------------

    print()
    print("=== ACCEPTED SUBMISSIONS ===")

    accepted = df[df["verdict"] == "OK"]

    accepted_per_problem = (
        accepted[
            accepted["contest_id"].notna()
            & accepted["problem_index"].notna()
        ]
        .groupby(
            ["handle", "contest_id", "problem_index"],
            dropna=False,
        )
        .size()
        .rename("accepted_count")
    )

    print(
        f"Accepted rows: {len(accepted):,}"
    )

    print(
        f"Unique accepted user-contest-problem combinations: "
        f"{len(accepted_per_problem):,}"
    )

    print(
        f"Problems accepted more than once: "
        f"{(accepted_per_problem > 1).sum():,}"
    )

    # ------------------------------------------------------------
    # Missing key fields
    # ------------------------------------------------------------

    print()
    print("=== MISSING VALUES ===")

    for column in [
        "submission_id",
        "contest_id",
        "problem_index",
        "problem_name",
        "problem_rating",
        "verdict",
        "submission_time",
    ]:
        print(
            f"{column:25s}: "
            f"{df[column].isna().sum():,}"
        )

    # ------------------------------------------------------------
    # Suspicious timestamps
    # ------------------------------------------------------------

    print()
    print("=== TIME RANGE ===")

    print("Earliest:", df["submission_time"].min())
    print("Latest:", df["submission_time"].max())

    # ------------------------------------------------------------
    # Verdict + rating
    # ------------------------------------------------------------

    print()
    print("=== ACCEPTED RATED PROBLEMS ===")

    accepted_rated = df[
        (df["verdict"] == "OK")
        & df["problem_rating"].notna()
    ]

    print(
        f"Accepted rated submissions: "
        f"{len(accepted_rated):,}"
    )

    print(
        f"Unique accepted rated user-problem submissions: "
        f"{accepted_rated[['handle', 'contest_id', 'problem_index']].drop_duplicates().shape[0]:,}"
    )


if __name__ == "__main__":
    main()