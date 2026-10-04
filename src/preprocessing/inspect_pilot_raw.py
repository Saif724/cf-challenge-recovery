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

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "pilot"
)

RATING_DIR = RAW_DIR / "rating_history"
SUBMISSION_DIR = RAW_DIR / "submissions"


def main():
    pilot = pd.read_parquet(PILOT_FILE)

    print("=" * 70)
    print("PILOT RAW DATA — COVERAGE INSPECTION")
    print("=" * 70)

    print()
    print("=== PILOT ===")
    print(f"Users: {len(pilot):,}")

    rating_counts = []
    submission_counts = []

    rating_first = []
    rating_last = []

    submission_first = []
    submission_last = []

    missing_rating = []
    missing_submission = []

    for handle in pilot["handle"]:
        rating_path = (
            RATING_DIR / f"{handle}.json"
        )

        submission_path = (
            SUBMISSION_DIR / f"{handle}.json"
        )

        # --------------------------------------------------
        # Rating history
        # --------------------------------------------------

        if rating_path.exists():
            with rating_path.open(
                "r",
                encoding="utf-8",
            ) as f:
                ratings = json.load(f)

            rating_counts.append(
                len(ratings)
            )

            if ratings:
                rating_times = [
                    pd.to_datetime(
                        x["ratingUpdateTimeSeconds"],
                        unit="s",
                        utc=True,
                    )
                    for x in ratings
                ]

                rating_first.append(
                    min(rating_times)
                )

                rating_last.append(
                    max(rating_times)
                )

        else:
            missing_rating.append(handle)

        # --------------------------------------------------
        # Submissions
        # --------------------------------------------------

        if submission_path.exists():
            with submission_path.open(
                "r",
                encoding="utf-8",
            ) as f:
                submissions = json.load(f)

            submission_counts.append(
                len(submissions)
            )

            if submissions:
                submission_times = [
                    pd.to_datetime(
                        x["creationTimeSeconds"],
                        unit="s",
                        utc=True,
                    )
                    for x in submissions
                ]

                submission_first.append(
                    min(submission_times)
                )

                submission_last.append(
                    max(submission_times)
                )

        else:
            missing_submission.append(handle)

    # ------------------------------------------------------
    # Counts
    # ------------------------------------------------------

    print()
    print("=== FILE COVERAGE ===")
    print(
        f"Rating files: "
        f"{len(rating_counts)}/{len(pilot)}"
    )

    print(
        f"Submission files: "
        f"{len(submission_counts)}/{len(pilot)}"
    )

    print(
        f"Missing rating files: "
        f"{len(missing_rating)}"
    )

    print(
        f"Missing submission files: "
        f"{len(missing_submission)}"
    )

    # ------------------------------------------------------
    # Rating history
    # ------------------------------------------------------

    if rating_counts:
        rating_series = pd.Series(
            rating_counts,
            name="rating_contests",
        )

        print()
        print("=== RATING HISTORY ===")
        print(
            rating_series.describe()
        )

        print()
        print("Rating contest quantiles:")
        print(
            rating_series.quantile(
                [0.10, 0.25, 0.50, 0.75, 0.90]
            )
        )

        print()
        print(
            "Users with >= 5 rated contests:",
            (rating_series >= 5).sum(),
        )

        print(
            "Users with >= 10 rated contests:",
            (rating_series >= 10).sum(),
        )

        print(
            "Users with >= 20 rated contests:",
            (rating_series >= 20).sum(),
        )

    # ------------------------------------------------------
    # Submission history
    # ------------------------------------------------------

    if submission_counts:
        submission_series = pd.Series(
            submission_counts,
            name="submissions",
        )

        print()
        print("=== SUBMISSIONS ===")
        print(
            submission_series.describe()
        )

        print()
        print("Submission quantiles:")
        print(
            submission_series.quantile(
                [0.10, 0.25, 0.50, 0.75, 0.90]
            )
        )

        print()
        print(
            "Users with >= 10 submissions:",
            (submission_series >= 10).sum(),
        )

        print(
            "Users with >= 50 submissions:",
            (submission_series >= 50).sum(),
        )

        print(
            "Users with >= 100 submissions:",
            (submission_series >= 100).sum(),
        )

        print(
            "Users with >= 500 submissions:",
            (submission_series >= 500).sum(),
        )

    # ------------------------------------------------------
    # Time coverage
    # ------------------------------------------------------

    if rating_first:
        print()
        print("=== RATING TIME COVERAGE ===")
        print(
            "Earliest:",
            min(rating_first),
        )
        print(
            "Latest:",
            max(rating_last),
        )

    if submission_first:
        print()
        print("=== SUBMISSION TIME COVERAGE ===")
        print(
            "Earliest:",
            min(submission_first),
        )
        print(
            "Latest:",
            max(submission_last),
        )

    # ------------------------------------------------------
    # User-level observation span
    # ------------------------------------------------------

    coverage_rows = []

    for handle in pilot["handle"]:
        rating_path = (
            RATING_DIR / f"{handle}.json"
        )

        submission_path = (
            SUBMISSION_DIR / f"{handle}.json"
        )

        if not (
            rating_path.exists()
            and submission_path.exists()
        ):
            continue

        with rating_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            ratings = json.load(f)

        with submission_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            submissions = json.load(f)

        if not ratings or not submissions:
            continue

        all_times = [
            pd.to_datetime(
                x["ratingUpdateTimeSeconds"],
                unit="s",
                utc=True,
            )
            for x in ratings
        ]

        all_times.extend(
            pd.to_datetime(
                x["creationTimeSeconds"],
                unit="s",
                utc=True,
            )
            for x in submissions
        )

        first_time = min(all_times)
        last_time = max(all_times)

        months = (
            (last_time.year - first_time.year) * 12
            + last_time.month
            - first_time.month
            + 1
        )

        coverage_rows.append(
            {
                "handle": handle,
                "first_activity": first_time,
                "last_activity": last_time,
                "coverage_months": months,
            }
        )

    coverage = pd.DataFrame(
        coverage_rows
    )

    if not coverage.empty:
        print()
        print("=== LONGITUDINAL COVERAGE ===")
        print(
            coverage["coverage_months"]
            .describe()
        )

        print()
        print("Coverage quantiles:")
        print(
            coverage["coverage_months"].quantile(
                [0.10, 0.25, 0.50, 0.75, 0.90]
            )
        )

        print()
        for threshold in [3, 6, 12, 18, 24, 30]:
            print(
                f"Users with >= {threshold} months:",
                (
                    coverage["coverage_months"]
                    >= threshold
                ).sum(),
            )


if __name__ == "__main__":
    main()