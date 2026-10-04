from pathlib import Path
import json
import time

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "contests.json"
)


def main():
    url = "https://codeforces.com/api/contest.list"

    print("Fetching Codeforces contest metadata...")

    response = requests.get(
        url,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()

    if data["status"] != "OK":
        raise RuntimeError(
            f"Codeforces API error: "
            f"{data.get('comment', 'unknown error')}"
        )

    contests = data["result"]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            contests,
            f,
            indent=2,
        )

    print(f"Saved {len(contests):,} contests.")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()