from pathlib import Path
import json

from client import CodeforcesClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    client = CodeforcesClient()

    print("Fetching contest list...")

    contests = client._request("contest.list")

    print()
    print("Total contests:", len(contests))

    if contests:
        print()
        print("Fields:")
        print(contests[0].keys())

    output = RAW_DIR / "contest_list.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        json.dump(contests, f, indent=2)

    print()
    print("Saved to:", output)


if __name__ == "__main__":
    main()