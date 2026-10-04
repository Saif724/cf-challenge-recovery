from pathlib import Path
import time

import pandas as pd

from client import CodeforcesClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_rating_diagnostic.parquet"
)

CANDIDATES = [
    "Benq",
    "jiangly",
    "Um_nik",
    "Errichto",
    "maroonrk",
    "Radewoosh",
    "ecnerwala",
]


def main():
    client = CodeforcesClient()

    rows = []

    for handle in CANDIDATES:
        print(f"\nFetching: {handle}")

        try:
            history = client.get_rating_history(handle)
        except Exception as exc:
            print(f"ERROR: {handle}: {exc}")
            continue

        if not history:
            print(f"No rating history: {handle}")
            continue

        history = sorted(
            history,
            key=lambda x: x["ratingUpdateTimeSeconds"],
        )

        for i, contest in enumerate(history, start=1):
            rows.append(
                {
                    "handle": handle,
                    "contest_number": i,
                    "contest_id": contest["contestId"],
                    "contest_name": contest["contestName"],
                    "rating_time": pd.to_datetime(
                        contest["ratingUpdateTimeSeconds"],
                        unit="s",
                        utc=True,
                    ),
                    "old_rating": contest["oldRating"],
                    "new_rating": contest["newRating"],
                }
            )

        print(
            f"Rated contests: {len(history)}"
        )

    if not rows:
        raise RuntimeError(
            "No candidate rating histories were collected."
        )

    df = pd.DataFrame(rows)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("CANDIDATE RATING DIAGNOSTIC")
    print("=" * 70)

    for handle in CANDIDATES:
        user = df[df["handle"] == handle].copy()

        if user.empty:
            continue

        first = user.iloc[0]
        peak_idx = user["new_rating"].idxmax()
        peak = user.loc[peak_idx]

        print(f"\n{handle}")
        print("-" * len(handle))

        print(
            f"First contest:      #{int(first['contest_id'])}"
        )
        print(
            f"First contest date: {first['rating_time']}"
        )
        print(
            f"First old rating:   {int(first['old_rating'])}"
        )
        print(
            f"First new rating:   {int(first['new_rating'])}"
        )
        print(
            f"Peak rating:        {int(peak['new_rating'])}"
        )
        print(
            f"Rated contests:     {len(user)}"
        )

        print("\nFirst 20 contests:")

        first20 = user.head(20)

        print(
            first20[
                [
                    "contest_number",
                    "contest_id",
                    "rating_time",
                    "old_rating",
                    "new_rating",
                ]
            ].to_string(index=False)
        )

    print(
        f"\nSaved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()