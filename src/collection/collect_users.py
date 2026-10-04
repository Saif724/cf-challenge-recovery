from pathlib import Path
import json

from client import CodeforcesClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

HANDLES = [
    "Benq",
    "jiangly",
    "ecnerwala",
]


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def collect_user(client: CodeforcesClient, handle: str):
    user_dir = RAW_DIR / handle
    user_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"COLLECTING: {handle}")
    print("=" * 70)

    print("Fetching user info...")
    user_info = client.get_user_info(handle)
    save_json(user_dir / "user_info.json", user_info)

    print("Fetching rating history...")
    rating_history = client.get_rating_history(handle)
    save_json(user_dir / "rating_history.json", rating_history)

    print(f"Rated contests: {len(rating_history)}")

    print("Fetching submissions...")

    all_submissions = []
    from_ = 1
    count = 1000

    while True:
        submissions = client.get_submissions(
            handle,
            from_=from_,
            count=count,
        )

        if not submissions:
            break

        all_submissions.extend(submissions)

        print(
            f"  fetched {len(submissions)} submissions "
            f"(total: {len(all_submissions)})"
        )

        if len(submissions) < count:
            break

        from_ += count

    save_json(
        user_dir / "submissions.json",
        all_submissions,
    )

    print(
        f"Saved {len(all_submissions)} submissions."
    )


def main():
    client = CodeforcesClient()

    for handle in HANDLES:
        try:
            collect_user(client, handle)
        except Exception as exc:
            print(f"\nERROR collecting {handle}: {exc}\n")


if __name__ == "__main__":
    main()