from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RATINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ratings.parquet"
)


def load_ratings():
    ratings = pd.read_parquet(RATINGS_FILE)

    ratings["rating_time"] = pd.to_datetime(
        ratings["rating_update_time_seconds"],
        unit="s",
        utc=True,
    )

    return ratings.sort_values("rating_time").reset_index(drop=True)


def rating_at_time(
    ratings: pd.DataFrame,
    timestamp: pd.Timestamp,
):
    """
    Return the latest official rating strictly before timestamp.

    Returns None if no rating update occurred before timestamp.
    """

    previous = ratings[
        ratings["rating_time"] < timestamp
    ]

    if previous.empty:
        return None

    return int(previous.iloc[-1]["new_rating"])


def main():
    ratings = load_ratings()

    print("=== RATING AT TIME TESTS ===")

    first_rating_time = ratings.iloc[0]["rating_time"]
    second_rating_time = ratings.iloc[1]["rating_time"]

    print("\nBefore first rating:")
    test_time = first_rating_time - pd.Timedelta(seconds=1)
    print(f"Timestamp: {test_time}")
    print(f"Rating: {rating_at_time(ratings, test_time)}")

    print("\nImmediately after first rating:")
    test_time = first_rating_time + pd.Timedelta(seconds=1)
    print(f"Timestamp: {test_time}")
    print(f"Rating: {rating_at_time(ratings, test_time)}")

    print("\nImmediately before second rating:")
    test_time = second_rating_time - pd.Timedelta(seconds=1)
    print(f"Timestamp: {test_time}")
    print(f"Rating: {rating_at_time(ratings, test_time)}")

    print("\nImmediately after second rating:")
    test_time = second_rating_time + pd.Timedelta(seconds=1)
    print(f"Timestamp: {test_time}")
    print(f"Rating: {rating_at_time(ratings, test_time)}")


if __name__ == "__main__":
    main()