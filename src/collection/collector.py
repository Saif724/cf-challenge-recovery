import json
from pathlib import Path

from client import CodeforcesClient


RAW_DIR = Path("../../data/raw")


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def collect_user(handle: str):
    client = CodeforcesClient()

    user_dir = RAW_DIR / handle
    user_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting data for {handle}...")

    # --------------------------------------------------
    # User information
    # --------------------------------------------------

    user_info = client.get_user_info(handle)

    save_json(
        user_info,
        user_dir / "user_info.json",
    )

    print("  ✓ user info")

    # --------------------------------------------------
    # Rating history
    # --------------------------------------------------

    rating_history = client.get_rating_history(handle)

    save_json(
        rating_history,
        user_dir / "rating_history.json",
    )

    print(f"  ✓ rating history ({len(rating_history)} records)")

    # --------------------------------------------------
    # Submissions
    # --------------------------------------------------

    all_submissions = []

    from_ = 1
    count = 1000

    while True:
        print(
            f"  fetching submissions "
            f"{from_} - {from_ + count - 1}..."
        )

        submissions = client.get_submissions(
            handle,
            from_=from_,
            count=count,
        )

        if not submissions:
            break

        all_submissions.extend(submissions)

        print(
            f"  received {len(submissions)} "
            f"(total: {len(all_submissions)})"
        )

        if len(submissions) < count:
            break

        from_ += count

    save_json(
        all_submissions,
        user_dir / "submissions.json",
    )

    print(
        f"  ✓ submissions ({len(all_submissions)} records)"
    )

    print(f"\nFinished collecting {handle}.")


if __name__ == "__main__":
    collect_user("tourist")