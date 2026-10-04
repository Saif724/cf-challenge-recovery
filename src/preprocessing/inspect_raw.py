import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "tourist"


def load_json(filename: str):
    path = RAW_DIR / filename

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def inspect_user_info():
    data = load_json("user_info.json")

    print("\n=== USER INFO ===")
    print("Records:", len(data))

    if data:
        print("Fields:")
        for field in data[0]:
            print(f"  - {field}")


def inspect_ratings():
    data = load_json("rating_history.json")

    print("\n=== RATING HISTORY ===")
    print("Records:", len(data))

    if data:
        print("Fields:")
        for field in data[0]:
            print(f"  - {field}")

        print("\nFirst record:")
        print(data[0])

        print("\nLast record:")
        print(data[-1])


def inspect_submissions():
    data = load_json("submissions.json")

    print("\n=== SUBMISSIONS ===")
    print("Records:", len(data))

    if not data:
        return

    print("\nTop-level fields:")
    for field in data[0]:
        print(f"  - {field}")

    print("\nProblem fields:")
    for field in data[0]["problem"]:
        print(f"  - {field}")

    print("\nAuthor fields:")
    for field in data[0]["author"]:
        print(f"  - {field}")

    verdicts = Counter(
        submission.get("verdict")
        for submission in data
    )

    participant_types = Counter(
        submission.get("author", {}).get("participantType")
        for submission in data
    )

    contests = {
        submission.get("contestId")
        for submission in data
        if submission.get("contestId") is not None
    }

    problem_ratings = [
        submission.get("problem", {}).get("rating")
        for submission in data
        if submission.get("problem", {}).get("rating") is not None
    ]

    print("\n=== DISTRIBUTIONS ===")

    print("\nVerdicts:")
    for verdict, count in verdicts.most_common():
        print(f"  {verdict}: {count}")

    print("\nParticipant types:")
    for participant_type, count in participant_types.most_common():
        print(f"  {participant_type}: {count}")

    print("\nDistinct contests:", len(contests))

    print(
        "Submissions with problem rating:",
        len(problem_ratings),
    )

    print(
        "Submissions without problem rating:",
        len(data) - len(problem_ratings),
    )

    timestamps = [
        submission["creationTimeSeconds"]
        for submission in data
        if submission.get("creationTimeSeconds") is not None
    ]

    print("\n=== TIME COVERAGE ===")

    print(
        "Newest timestamp:",
        max(timestamps),
    )

    print(
        "Oldest timestamp:",
        min(timestamps),
    )


def main():
    inspect_user_info()
    inspect_ratings()
    inspect_submissions()


if __name__ == "__main__":
    main()