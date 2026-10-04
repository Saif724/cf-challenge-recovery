from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "submissions.parquet"
)


def main():
    df = pd.read_parquet(INPUT_FILE)

    print("=== BASIC DATA ===")
    print(f"Total submissions: {len(df):,}")
    print(f"Unique submission IDs: {df['submission_id'].nunique():,}")
    print(f"Unique problems attempted: "
          f"{df[['contest_id', 'problem_index']].drop_duplicates().shape[0]:,}")

    print("\n=== VERDICTS ===")
    print(df["verdict"].value_counts(dropna=False).to_string())

    accepted = df[df["verdict"] == "OK"].copy()

    print("\n=== ACCEPTED SUBMISSIONS ===")
    print(f"Accepted submissions: {len(accepted):,}")

    print("\nProblem-rating availability:")
    print(
        accepted["problem_rating"]
        .notna()
        .value_counts()
        .rename({
            True: "with_rating",
            False: "without_rating",
        })
        .to_string()
    )

    accepted_with_rating = accepted[
        accepted["problem_rating"].notna()
    ].copy()

    accepted_without_rating = accepted[
        accepted["problem_rating"].isna()
    ].copy()

    print("\n=== ACCEPTED RATING COVERAGE ===")

    total_accepted = len(accepted)

    if total_accepted > 0:
        coverage = len(accepted_with_rating) / total_accepted * 100
        missing = len(accepted_without_rating) / total_accepted * 100

        print(f"Accepted with rating: "
              f"{len(accepted_with_rating):,} ({coverage:.2f}%)")

        print(f"Accepted without rating: "
              f"{len(accepted_without_rating):,} ({missing:.2f}%)")

    print("\n=== UNIQUE ACCEPTED PROBLEMS ===")

    accepted_problems = (
        accepted[
            ["contest_id", "problem_index", "problem_name", "problem_rating"]
        ]
        .drop_duplicates(
            subset=["contest_id", "problem_index"]
        )
    )

    print(
        f"Unique accepted problems: "
        f"{len(accepted_problems):,}"
    )

    rated_accepted_problems = accepted_problems[
        accepted_problems["problem_rating"].notna()
    ]

    unrated_accepted_problems = accepted_problems[
        accepted_problems["problem_rating"].isna()
    ]

    print(
        f"Rated accepted problems: "
        f"{len(rated_accepted_problems):,}"
    )

    print(
        f"Unrated accepted problems: "
        f"{len(unrated_accepted_problems):,}"
    )

    print("\n=== PARTICIPANT TYPES ===")

    print(
        accepted["participant_type"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n=== ACCEPTED PROBLEMS BY PARTICIPANT TYPE ===")

    print(
        accepted.groupby("participant_type")
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\n=== PROBLEM RATINGS ===")

    if len(rated_accepted_problems) > 0:
        print(
            rated_accepted_problems["problem_rating"]
            .describe()
            .to_string()
        )

    print("\n=== ACCEPTED PROBLEMS WITHOUT RATINGS ===")

    if len(unrated_accepted_problems) > 0:
        print(
            unrated_accepted_problems[
                [
                    "contest_id",
                    "problem_index",
                    "problem_name",
                ]
            ]
            .to_string(index=False)
        )
    else:
        print("None.")


if __name__ == "__main__":
    main()