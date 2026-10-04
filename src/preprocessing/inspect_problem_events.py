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

    df["submission_time"] = pd.to_datetime(
        df["creation_time_seconds"],
        unit="s",
        utc=True,
    )

    df = df.sort_values("submission_time")

    problem_keys = [
        "contest_id",
        "problem_index",
    ]

    grouped = (
        df.groupby(problem_keys, dropna=False)
        .agg(
            submissions=("submission_id", "count"),
            accepted=("verdict", lambda x: (x == "OK").sum()),
            first_submission=("submission_time", "min"),
            last_submission=("submission_time", "max"),
            participant_types=(
                "participant_type",
                lambda x: ", ".join(sorted(set(x.dropna())))
            ),
            problem_rating=("problem_rating", "first"),
            problem_name=("problem_name", "first"),
        )
        .reset_index()
    )

    print("=== PROBLEM-LEVEL EVENTS ===")
    print(f"Unique problem events: {len(grouped):,}")

    print("\n=== SUBMISSIONS PER PROBLEM ===")

    print(
        grouped["submissions"]
        .describe()
        .to_string()
    )

    print("\n=== ACCEPTANCE PATTERN ===")

    print(
        grouped["accepted"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\n=== PROBLEMS WITH MULTIPLE ACCEPTED SUBMISSIONS ===")

    multiple_accepted = grouped[
        grouped["accepted"] > 1
    ]

    print(
        f"Count: {len(multiple_accepted):,}"
    )

    if not multiple_accepted.empty:
        print(
            multiple_accepted[
                [
                    "contest_id",
                    "problem_index",
                    "problem_name",
                    "submissions",
                    "accepted",
                    "participant_types",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    print("\n=== PARTICIPANT TYPE COMBINATIONS ===")

    print(
        grouped["participant_types"]
        .value_counts()
        .head(30)
        .to_string()
    )

    print("\n=== SAMPLE MULTI-CONTEXT PROBLEMS ===")

    multi_context = grouped[
        grouped["participant_types"].str.contains(",", regex=False)
    ]

    print(
        f"Problems appearing under multiple participant types: "
        f"{len(multi_context):,}"
    )

    if not multi_context.empty:
        print(
            multi_context[
                [
                    "contest_id",
                    "problem_index",
                    "problem_name",
                    "submissions",
                    "accepted",
                    "participant_types",
                    "first_submission",
                    "last_submission",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()