import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ratings.parquet"
)


def load_rating_history(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize(handle: str, rating_history):
    rows = []

    for record in rating_history:
        rows.append(
            {
                "handle": handle,
                "contest_id": record.get("contestId"),
                "contest_name": record.get("contestName"),
                "contest_rank": record.get("rank"),
                "rating_update_time_seconds": record.get(
                    "ratingUpdateTimeSeconds"
                ),
                "old_rating": record.get("oldRating"),
                "new_rating": record.get("newRating"),
            }
        )

    return rows


def main():
    user_dirs = sorted(
        path
        for path in RAW_DIR.iterdir()
        if path.is_dir()
        and (path / "rating_history.json").exists()
    )

    if not user_dirs:
        raise RuntimeError(
            "No user rating histories found."
        )

    print(
        f"Found {len(user_dirs)} users:"
    )

    for user_dir in user_dirs:
        print(f"  - {user_dir.name}")

    all_rows = []

    print()

    for user_dir in user_dirs:
        handle = user_dir.name
        rating_file = user_dir / "rating_history.json"

        try:
            rating_history = load_rating_history(
                rating_file
            )
        except Exception as exc:
            print(
                f"{handle}: ERROR loading rating history: "
                f"{exc}"
            )
            continue

        rows = normalize(
            handle,
            rating_history,
        )

        all_rows.extend(rows)

        print(
            f"{handle}: "
            f"loaded {len(rating_history)} records, "
            f"normalized {len(rows)}"
        )

    if not all_rows:
        raise RuntimeError(
            "No rating records were normalized."
        )

    df = pd.DataFrame(all_rows)

    # ---------------------------------------------------------
    # Basic cleanup
    # ---------------------------------------------------------

    df["contest_id"] = pd.to_numeric(
        df["contest_id"],
        errors="coerce",
    ).astype("Int64")

    df["contest_rank"] = pd.to_numeric(
        df["contest_rank"],
        errors="coerce",
    ).astype("Int64")

    df["rating_update_time_seconds"] = pd.to_numeric(
        df["rating_update_time_seconds"],
        errors="coerce",
    ).astype("Int64")

    df["old_rating"] = pd.to_numeric(
        df["old_rating"],
        errors="coerce",
    )

    df["new_rating"] = pd.to_numeric(
        df["new_rating"],
        errors="coerce",
    )

    # ---------------------------------------------------------
    # Sort
    # ---------------------------------------------------------

    df = df.sort_values(
        [
            "handle",
            "rating_update_time_seconds",
            "contest_id",
        ]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Integrity checks
    # ---------------------------------------------------------

    duplicate_mask = df.duplicated(
        subset=[
            "handle",
            "contest_id",
        ],
        keep=False,
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    if duplicate_count:
        print(
            f"\nWARNING: {duplicate_count} rows belong "
            "to duplicated handle/contest_id pairs."
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("NORMALIZED RATINGS")
    print("=" * 70)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Users: {df['handle'].nunique():,}"
    )

    print("\nRecords per user:")

    print(
        df["handle"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nColumns:")

    print(
        df.columns.tolist()
    )

    print(
        f"\nSaved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()