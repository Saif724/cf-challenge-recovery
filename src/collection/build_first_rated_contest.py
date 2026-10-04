from pathlib import Path
import json

import pandas as pd
import requests

from client import CodeforcesClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONTEST_FILE = PROJECT_ROOT / "data" / "raw" / "contest_list.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "rated_participation"
STATUS_FILE = RAW_DIR / "_status.json"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "first_rated_contest.parquet"

START_YEAR = 2010
END_YEAR = 2024


def load_contests():
    with CONTEST_FILE.open("r", encoding="utf-8") as f:
        contests = json.load(f)

    df = pd.DataFrame(contests)

    df["start_time"] = pd.to_datetime(
        df["startTimeSeconds"],
        unit="s",
        utc=True,
    )

    df = df[
        (df["start_time"].dt.year >= START_YEAR)
        & (df["start_time"].dt.year <= END_YEAR)
    ].copy()

    return df.sort_values(
        ["start_time", "id"]
    ).reset_index(drop=True)


def load_status():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not STATUS_FILE.exists():
        return {}

    with STATUS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_status(status):
    with STATUS_FILE.open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)


def save_rating_changes(contest_id, changes):
    path = RAW_DIR / f"{contest_id}.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(changes, f)


def build_first_rated():
    with CONTEST_FILE.open("r", encoding="utf-8") as f:
        contests = json.load(f)

    contest_df = pd.DataFrame(contests)

    contest_df["start_time"] = pd.to_datetime(
        contest_df["startTimeSeconds"],
        unit="s",
        utc=True,
    )

    contest_df = contest_df[
        (contest_df["start_time"].dt.year >= START_YEAR)
        & (contest_df["start_time"].dt.year <= END_YEAR)
    ].copy()

    contest_df = contest_df.sort_values(
        ["start_time", "id"]
    )

    contest_info = {
        int(row["id"]): {
            "contest_name": row["name"],
            "contest_start_time": row["start_time"],
        }
        for _, row in contest_df.iterrows()
    }

    first = {}

    files = sorted(
        (
            p for p in RAW_DIR.glob("*.json")
            if p.stem.isdigit()
        ),
        key=lambda p: contest_info.get(
            int(p.stem),
            {}
        ).get(
            "contest_start_time",
            pd.Timestamp.max.tz_localize("UTC"),
        ),
    )

    for path in files:
        contest_id = int(path.stem)

        if contest_id not in contest_info:
            continue

        with path.open("r", encoding="utf-8") as f:
            changes = json.load(f)

        contest = contest_info[contest_id]

        for change in changes:
            handle = change["handle"]

            if handle not in first:
                first[handle] = {
                    "handle": handle,
                    "first_contest_id": contest_id,
                    "first_contest_name": contest["contest_name"],
                    "first_contest_start_time": contest["contest_start_time"],
                    "first_rating_time": pd.to_datetime(
                        change["ratingUpdateTimeSeconds"],
                        unit="s",
                        utc=True,
                    ),
                    "initial_rating": change["newRating"],
                }

    if not first:
        return pd.DataFrame()

    df = pd.DataFrame(first.values())

    return df.sort_values(
        "first_contest_start_time"
    ).reset_index(drop=True)


def main():
    contests = load_contests()
    status = load_status()

    print("=" * 70)
    print("FIRST RATED CONTEST DISCOVERY")
    print("=" * 70)
    print(f"Contest candidates: {len(contests)}")

    completed = sum(
        1
        for value in status.values()
        if value.get("status") in {"rated", "unrated"}
    )

    print(f"Completed: {completed}")

    client = CodeforcesClient()

    for i, contest in contests.iterrows():
        contest_id = int(contest["id"])
        name = contest["name"]

        contest_key = str(contest_id)

        existing = status.get(contest_key)

        if existing and existing["status"] in {"rated", "unrated"}:
            continue

        print()
        print("=" * 70)
        print(
            f"[{i + 1}/{len(contests)}] "
            f"{contest_id} — {name}"
        )

        # Explicitly unrated contests never need an API request.
        if "unrated" in name.lower():
            print("SKIP: explicitly unrated")

            status[contest_key] = {
                "status": "unrated",
                "name": name,
            }

            save_status(status)
            continue

        try:
            changes = client._request(
                "contest.ratingChanges",
                {"contestId": contest_id},
            )

            save_rating_changes(contest_id, changes)

            if changes:
                result = "rated"
                print(
                    f"RATED — {len(changes):,} rating changes"
                )
            else:
                result = "unrated"
                print("NO RATING CHANGES")

            status[contest_key] = {
                "status": result,
                "name": name,
                "records": len(changes),
            }

            save_status(status)

        except requests.HTTPError as exc:
            response = exc.response

            if response is not None and response.status_code == 400:
                print("UNRATED / INVALID FOR RATING CHANGES")

                status[contest_key] = {
                    "status": "unrated",
                    "name": name,
                    "http_status": 400,
                }

                save_status(status)

            else:
                print(f"TEMPORARY HTTP ERROR: {exc}")
                print("Contest will be retried next run.")

        except Exception as exc:
            print(f"TEMPORARY ERROR: {exc}")
            print("Contest will be retried next run.")

    print()
    print("=" * 70)
    print("BUILDING FIRST-RATED-CONTEST TABLE")
    print("=" * 70)

    df = build_first_rated()

    if df.empty:
        print("No rated participation data found.")
        return

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        OUTPUT,
        index=False,
    )

    print()
    print("=== RESULT ===")
    print(f"Users: {len(df):,}")
    print(f"Output: {OUTPUT}")

    print()
    print("=== FIRST RATED CONTEST YEARS ===")
    print(
        df["first_rating_time"]
        .dt.year
        .value_counts()
        .sort_index()
    )

    print()
    print("=== TARGET COHORT: 2022–2024 ===")

    cohort = df[
        df["first_rating_time"].dt.year.between(2022, 2024)
    ]

    print(f"Eligible users: {len(cohort):,}")


if __name__ == "__main__":
    main()