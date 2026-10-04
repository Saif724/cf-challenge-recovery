from pathlib import Path
import json
import time

import pandas as pd

from client import CodeforcesAPIError, CodeforcesClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_sample_200.parquet"
)

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "pilot"
)

RATING_DIR = RAW_DIR / "rating_history"
SUBMISSION_DIR = RAW_DIR / "submissions"

STATUS_FILE = RAW_DIR / "_status.json"

SUBMISSIONS_PER_REQUEST = 1000


def load_status():
    if not STATUS_FILE.exists():
        return {}

    with STATUS_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_status(status):
    with STATUS_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            status,
            f,
            indent=2,
        )


def collect_rating_history(
    client,
    handle,
    output_path,
):
    result = client.get_rating_history(handle)

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    return len(result)


def collect_all_submissions(
    client,
    handle,
    output_path,
):
    all_submissions = []
    from_ = 1

    while True:
        submissions = client.get_submissions(
            handle,
            from_=from_,
            count=SUBMISSIONS_PER_REQUEST,
        )

        all_submissions.extend(submissions)

        if len(submissions) < SUBMISSIONS_PER_REQUEST:
            break

        from_ += SUBMISSIONS_PER_REQUEST

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            all_submissions,
            f,
            indent=2,
        )

    return len(all_submissions)


def main():
    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RATING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUBMISSION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pilot = pd.read_parquet(INPUT)

    handles = (
        pilot["handle"]
        .drop_duplicates()
        .tolist()
    )

    status = load_status()

    client = CodeforcesClient()

    print("=" * 70)
    print("CODEFORCES PILOT DATA COLLECTION")
    print("=" * 70)

    print()
    print(f"Users: {len(handles):,}")

    completed = 0

    for index, handle in enumerate(handles, start=1):
        print()
        print(
            f"[{index}/{len(handles)}] {handle}"
        )

        user_status = status.get(
            handle,
            {},
        )

        # --------------------------------------------------
        # Rating history
        # --------------------------------------------------

        rating_path = (
            RATING_DIR
            / f"{handle}.json"
        )

        if user_status.get(
            "rating_history"
        ) == "completed" and rating_path.exists():

            print("  rating: already collected")

        else:
            try:
                count = collect_rating_history(
                    client,
                    handle,
                    rating_path,
                )

                user_status[
                    "rating_history"
                ] = "completed"

                user_status[
                    "rating_count"
                ] = count

                status[handle] = user_status
                save_status(status)

                print(
                    f"  rating: {count} records"
                )

            except Exception as exc:
                user_status[
                    "rating_history"
                ] = "error"

                user_status[
                    "rating_error"
                ] = str(exc)

                status[handle] = user_status
                save_status(status)

                print(
                    f"  rating ERROR: {exc}"
                )

                continue

        # --------------------------------------------------
        # Submissions
        # --------------------------------------------------

        submission_path = (
            SUBMISSION_DIR
            / f"{handle}.json"
        )

        if user_status.get(
            "submissions"
        ) == "completed" and submission_path.exists():

            print(
                "  submissions: already collected"
            )

        else:
            try:
                count = collect_all_submissions(
                    client,
                    handle,
                    submission_path,
                )

                user_status[
                    "submissions"
                ] = "completed"

                user_status[
                    "submission_count"
                ] = count

                status[handle] = user_status
                save_status(status)

                print(
                    f"  submissions: {count} records"
                )

            except Exception as exc:
                user_status[
                    "submissions"
                ] = "error"

                user_status[
                    "submission_error"
                ] = str(exc)

                status[handle] = user_status
                save_status(status)

                print(
                    f"  submissions ERROR: {exc}"
                )

                continue

        completed += 1

    print()
    print("=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)

    print(
        f"Completed users: {completed}/{len(handles)}"
    )

    print(
        f"Raw data: {RAW_DIR}"
    )


if __name__ == "__main__":
    main()