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

RATING_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "pilot"
    / "rating_history"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_FILE = OUTPUT_DIR / "pilot_rating_history.parquet"


def normalize_rating(handle: str, row: dict) -> dict:
    contest_time = row.get("ratingUpdateTimeSeconds")

    if contest_time is not None:
        contest_datetime = pd.to_datetime(
            contest_time,
            unit="s",
            utc=True,
        )
    else:
        contest_datetime = pd.NaT

    return {
        "handle": handle,
        "contest_id": row.get("contestId"),
        "contest_name": row.get("contestName"),
        "rank": row.get("rank"),
        "old_rating": row.get("oldRating"),
        "new_rating": row.get("newRating"),
        "rating_change": row.get("newRating", 0)
        - row.get("oldRating", 0),
        "contest_time": contest_datetime,
        "contest_time_seconds": contest_time,
    }


def main():
    pilot = pd.read_parquet(PILOT_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    print("=" * 70)
    print("NORMALIZING PILOT RATING HISTORY")
    print("=" * 70)

    for i, handle in enumerate(pilot["handle"], start=1):
        path = RATING_DIR / f"{handle}.json"

        with path.open("r", encoding="utf-8") as f:
            ratings = json.load(f)

        for row in ratings:
            rows.append(
                normalize_rating(
                    handle,
                    row,
                )
            )

        if i % 25 == 0 or i == len(pilot):
            print(f"Processed: {i}/{len(pilot)}")

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            ["handle", "contest_time", "contest_id"]
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
    print(f"Unique contests: {df['contest_id'].nunique():,}")

    print()
    print("=== TIME RANGE ===")
    print("Earliest:", df["contest_time"].min())
    print("Latest:", df["contest_time"].max())

    print()
    print("=== RATING ===")
    print(
        "Initial rating:",
        df.groupby("handle")["old_rating"].first().describe()
    )

    print()
    print("=== RATING CHANGE ===")
    print(df["rating_change"].describe())

    print()
    print("=== CONTEST PARTICIPATION ===")

    contests_per_user = (
        df.groupby("handle")["contest_id"]
        .nunique()
    )

    print(contests_per_user.describe())

    print()
    print("Users with >= 5 contests:",
          (contests_per_user >= 5).sum())

    print("Users with >= 10 contests:",
          (contests_per_user >= 10).sum())

    print("Users with >= 20 contests:",
          (contests_per_user >= 20).sum())

    print()
    print("Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()