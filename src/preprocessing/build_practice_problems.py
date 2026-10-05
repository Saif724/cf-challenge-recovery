from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_submission_activity.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_practice_problems.parquet"
)


def main():
    submissions = pd.read_parquet(INPUT_FILE)

    print("=" * 70)
    print("BUILDING ACCEPTED PRACTICE PROBLEMS")
    print("=" * 70)

    # ------------------------------------------------------------
    # Only accepted rated submissions
    # ------------------------------------------------------------

    accepted = submissions[
        submissions["accepted"]
        & submissions["rated_problem"]
    ].copy()

    # ------------------------------------------------------------
    # CLEAN PRACTICE
    #
    # Only ordinary practice submissions.
    # ------------------------------------------------------------

    clean = accepted[
        accepted["activity_type"] == "practice"
    ].copy()

    # ------------------------------------------------------------
    # BROAD PRACTICE
    #
    # Ordinary practice + post-contest activity.
    # ------------------------------------------------------------

    broad = accepted[
        accepted["activity_type"].isin(
            ["practice", "post_contest"]
        )
    ].copy()

    # ------------------------------------------------------------
    # Collapse repeated accepted submissions.
    #
    # A user/problem should contribute once.
    #
    # Keep the first accepted timestamp in each activity
    # category.
    # ------------------------------------------------------------

    keys = [
        "handle",
        "contest_id",
        "problem_index",
        "problem_rating",
        "problem_name",
    ]

    clean_problems = (
        clean
        .sort_values("submission_time")
        .drop_duplicates(
            subset=keys,
            keep="first",
        )
        .copy()
    )

    broad_problems = (
        broad
        .sort_values("submission_time")
        .drop_duplicates(
            subset=keys,
            keep="first",
        )
        .copy()
    )

    # ------------------------------------------------------------
    # Label the datasets
    # ------------------------------------------------------------

    clean_problems["practice_definition"] = "clean"
    broad_problems["practice_definition"] = "broad"

    # ------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------

    result = pd.concat(
        [
            clean_problems,
            broad_problems,
        ],
        ignore_index=True,
    )

    # ------------------------------------------------------------
    # Practice month
    # ------------------------------------------------------------

    result["practice_month"] = (
        result["submission_time"]
        .dt.tz_convert(None)
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    # ------------------------------------------------------------
    # Save only useful columns
    # ------------------------------------------------------------

    columns = [
        "handle",
        "contest_id",
        "problem_index",
        "problem_name",
        "problem_rating",
        "submission_time",
        "practice_month",
        "practice_definition",
    ]

    result = result[columns]

    result.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    print()
    print("Clean accepted practice problems:")
    print(len(clean_problems))

    print()
    print("Broad accepted practice problems:")
    print(len(broad_problems))

    print()
    print("Users:")
    print(result["handle"].nunique())

    print()
    print("=== PROBLEMS BY DEFINITION ===")
    print(
        result["practice_definition"]
        .value_counts()
    )

    print()
    print("=== MONTHS ===")
    print(
        result["practice_month"]
        .nunique()
    )

    print()
    print("=== DIFFICULTY SUMMARY ===")

    difficulty_summary = (
        result
        .groupby("practice_definition")["problem_rating"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            p75=lambda x: x.quantile(0.75),
            p90=lambda x: x.quantile(0.90),
            min="min",
            max="max",
        )
    )

    print(difficulty_summary)

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()