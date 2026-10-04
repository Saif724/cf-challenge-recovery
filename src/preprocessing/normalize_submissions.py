from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PILOT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_sample_200.parquet"
)

SUBMISSION_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "pilot"
    / "submissions"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_FILE = OUTPUT_DIR / "pilot_submissions.parquet"


def normalize_submission(handle: str, submission: dict) -> dict:
    problem = submission.get("problem", {})

    creation_time = submission.get("creationTimeSeconds")

    if creation_time is not None:
        timestamp = pd.to_datetime(
            creation_time,
            unit="s",
            utc=True,
        )
    else:
        timestamp = pd.NaT

    return {
        "handle": handle,
        "submission_id": submission.get("id"),
        "contest_id": submission.get("contestId"),
        "problem_index": problem.get("index"),
        "problem_name": problem.get("name"),
        "problem_rating": problem.get("rating"),
        "problem_tags": problem.get("tags"),
        "verdict": submission.get("verdict"),
        "programming_language": submission.get("programmingLanguage"),
        "submission_time": timestamp,
        "submission_time_seconds": creation_time,
    }


def main():
    pilot = pd.read_parquet(PILOT_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    print("=" * 70)
    print("NORMALIZING PILOT SUBMISSIONS")
    print("=" * 70)

    for i, handle in enumerate(pilot["handle"], start=1):
        path = SUBMISSION_DIR / f"{handle}.json"

        with path.open("r", encoding="utf-8") as f:
            submissions = json.load(f)

        for submission in submissions:
            rows.append(
                normalize_submission(
                    handle,
                    submission,
                )
            )

        if i % 25 == 0 or i == len(pilot):
            print(f"Processed: {i}/{len(pilot)}")

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            ["handle", "submission_time", "submission_id"]
        ).reset_index(drop=True)

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("NORMALIZATION COMPLETE")
    print("=" * 70)

    print(f"Rows: {len(df):,}")
    print(f"Users: {df['handle'].nunique():,}")

    print()
    print("=== VERDICTS ===")
    print(df["verdict"].value_counts(dropna=False).head(20))

    print()
    print("=== RATED PROBLEMS ===")
    print(
        f"Rated problem submissions: "
        f"{df['problem_rating'].notna().sum():,}"
    )
    print(
        f"Unrated problem submissions: "
        f"{df['problem_rating'].isna().sum():,}"
    )

    print()
    print("=== CONTEST IDS ===")
    print(
        f"Contest submissions: "
        f"{df['contest_id'].notna().sum():,}"
    )
    print(
        f"Non-contest submissions: "
        f"{df['contest_id'].isna().sum():,}"
    )

    print()
    print("=== ACCEPTED ===")
    print(
        f"Accepted submissions: "
        f"{(df['verdict'] == 'OK').sum():,}"
    )

    print()
    print("Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()