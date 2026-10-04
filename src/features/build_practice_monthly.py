from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUBMISSIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "submissions.parquet"
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
    / "derived"
    / "practice_monthly.parquet"
)


def main():
    submissions = pd.read_parquet(SUBMISSIONS_FILE)

    with open(CONTESTS_FILE, encoding="utf-8") as f:
        contests = pd.DataFrame(json.load(f))

    # ---------------------------------------------------------
    # Submission timestamp
    # ---------------------------------------------------------

    submissions["submission_time"] = pd.to_datetime(
        submissions["creation_time_seconds"],
        unit="s",
        utc=True,
    )

    # ---------------------------------------------------------
    # Official contest boundaries
    # ---------------------------------------------------------

    contests["contest_end_time_seconds"] = (
        contests["startTimeSeconds"]
        + contests["durationSeconds"]
    )

    contests = contests[
        [
            "id",
            "startTimeSeconds",
            "contest_end_time_seconds",
        ]
    ]

    submissions = submissions.merge(
        contests,
        left_on="contest_id",
        right_on="id",
        how="left",
    )

    # ---------------------------------------------------------
    # Determine whether submission happened during
    # the official contest.
    # ---------------------------------------------------------

    submissions["during_contest"] = (
        submissions["startTimeSeconds"].notna()
        &
        (
            submissions["creation_time_seconds"]
            >= submissions["startTimeSeconds"]
        )
        &
        (
            submissions["creation_time_seconds"]
            < submissions["contest_end_time_seconds"]
        )
    )

    # ---------------------------------------------------------
    # Practice submission:
    #
    # Outside official contest window.
    #
    # For difficulty analysis we require a known problem rating.
    # ---------------------------------------------------------

    practice = submissions[
        (~submissions["during_contest"])
        &
        (submissions["problem_rating"].notna())
    ].copy()

    practice["month"] = (
        practice["submission_time"]
        .dt.strftime("%Y-%m")
    )

    # ---------------------------------------------------------
    # Stable problem identity
    #
    # Codeforces contest problems are identified by
    # (contest_id, problem_index).
    #
    # For submissions without a contest_id, include the
    # problem name as an additional identifier.
    # ---------------------------------------------------------

    practice["problem_key"] = (
        practice["contest_id"]
        .astype("Int64")
        .astype(str)
        + ":"
        + practice["problem_index"].astype(str)
        + ":"
        + practice["problem_name"].fillna("").astype(str)
    )

    # ---------------------------------------------------------
    # Attempted rated problems
    #
    # Count DISTINCT problems, not submissions.
    # ---------------------------------------------------------

    attempted_unique = (
        practice
        .drop_duplicates(
            [
                "handle",
                "month",
                "problem_key",
            ]
        )
    )

    attempted = (
        attempted_unique
        .groupby(
            ["handle", "month"],
            as_index=False,
        )
        .agg(
            attempted_rated_count=(
                "problem_key",
                "count",
            ),
            active_days=(
                "submission_time",
                lambda x: x.dt.date.nunique(),
            ),
        )
    )

    # ---------------------------------------------------------
    # Accepted rated problems
    #
    # Count DISTINCT problems with at least one accepted
    # submission during the month.
    # ---------------------------------------------------------

    accepted = practice[
        practice["verdict"] == "OK"
    ].copy()

    accepted_unique = (
        accepted
        .drop_duplicates(
            [
                "handle",
                "month",
                "problem_key",
            ]
        )
    )

    accepted_features = (
        accepted_unique
        .groupby(
            ["handle", "month"],
            as_index=False,
        )
        .agg(
            accepted_rated_count=(
                "problem_key",
                "count",
            ),
            median_accepted_rating=(
                "problem_rating",
                "median",
            ),
            mean_accepted_rating=(
                "problem_rating",
                "mean",
            ),
            p75_accepted_rating=(
                "problem_rating",
                lambda x: np.percentile(x, 75),
            ),
            p90_accepted_rating=(
                "problem_rating",
                lambda x: np.percentile(x, 90),
            ),
        )
    )

    # ---------------------------------------------------------
    # Combine
    # ---------------------------------------------------------

    monthly = attempted.merge(
        accepted_features,
        on=["handle", "month"],
        how="left",
    )

    monthly = monthly.sort_values(
        ["handle", "month"]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    monthly.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    print("=== PRACTICE MONTHLY ===")

    print(f"Rows: {len(monthly):,}")
    print(f"Users: {monthly['handle'].nunique():,}")
    print(f"Months: {monthly['month'].nunique():,}")

    print("\n=== COLUMNS ===")

    print(
        monthly.columns.tolist()
    )

    print("\n=== ATTEMPTED PROBLEM COUNTS ===")

    print(
        monthly["attempted_rated_count"]
        .describe()
        .to_string()
    )

    print("\n=== ACCEPTED PROBLEM COUNTS ===")

    print(
        monthly["accepted_rated_count"]
        .describe()
        .to_string()
    )

    print("\n=== MONTHLY SAMPLE ===")

    print(
        monthly.head(20).to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()