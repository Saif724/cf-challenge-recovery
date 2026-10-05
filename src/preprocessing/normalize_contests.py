from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONTEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "contest_list.json"
)

RATING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_rating_history.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_FILE = OUTPUT_DIR / "pilot_contests.parquet"


def main():
    print("=" * 70)
    print("NORMALIZING PILOT CONTEST METADATA")
    print("=" * 70)

    with CONTEST_FILE.open("r", encoding="utf-8") as f:
        contests = json.load(f)

    ratings = pd.read_parquet(RATING_FILE)

    # Only contests actually participated in by pilot users.
    pilot_contest_ids = set(
        ratings["contest_id"]
        .dropna()
        .astype(int)
        .unique()
    )

    print()
    print(f"All contests in contest_list.json: {len(contests):,}")
    print(
        f"Contests appearing in pilot rating histories: "
        f"{len(pilot_contest_ids):,}"
    )

    rows = []

    for contest in contests:
        contest_id = contest.get("id")

        if contest_id not in pilot_contest_ids:
            continue

        start_seconds = contest.get("startTimeSeconds")
        duration_seconds = contest.get("durationSeconds")

        if start_seconds is not None:
            start_time = pd.to_datetime(
                start_seconds,
                unit="s",
                utc=True,
            )
        else:
            start_time = pd.NaT

        if (
            start_seconds is not None
            and duration_seconds is not None
        ):
            end_time = pd.to_datetime(
                start_seconds + duration_seconds,
                unit="s",
                utc=True,
            )
        else:
            end_time = pd.NaT

        rows.append(
            {
                "contest_id": contest_id,
                "contest_name": contest.get("name"),
                "contest_type": contest.get("type"),
                "phase": contest.get("phase"),
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": duration_seconds,
                "duration_hours": (
                    duration_seconds / 3600
                    if duration_seconds is not None
                    else None
                ),
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            ["start_time", "contest_id"]
        ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("NORMALIZATION COMPLETE")
    print("=" * 70)

    print(f"Rows: {len(df):,}")
    print(f"Unique contests: {df['contest_id'].nunique():,}")

    print()
    print("=== MISSING VALUES ===")

    for column in [
        "contest_id",
        "contest_name",
        "contest_type",
        "phase",
        "start_time",
        "end_time",
        "duration_seconds",
    ]:
        print(
            f"{column:20s}: "
            f"{df[column].isna().sum():,}"
        )

    print()
    print("=== CONTEST TYPES ===")
    print(df["contest_type"].value_counts(dropna=False))

    print()
    print("=== PHASES ===")
    print(df["phase"].value_counts(dropna=False))

    print()
    print("=== TIME RANGE ===")
    print("Earliest:", df["start_time"].min())
    print("Latest:", df["start_time"].max())

    print()
    print("=== DURATIONS ===")
    print(df["duration_hours"].describe())

    print()
    print("Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()