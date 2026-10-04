from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT = PROJECT_ROOT / "data" / "raw" / "contest_list.json"


def main():
    with INPUT.open("r", encoding="utf-8") as f:
        contests = json.load(f)

    df = pd.DataFrame(contests)

    df["start_time"] = pd.to_datetime(
        df["startTimeSeconds"],
        unit="s",
        utc=True,
    )

    df["year"] = df["start_time"].dt.year

    print("=== TOTAL ===")
    print("Contests:", len(df))

    print()
    print("=== CONTEST TYPES ===")
    print(df["type"].value_counts())

    print()
    print("=== PHASES ===")
    print(df["phase"].value_counts())

    print()
    print("=== CONTESTS BY YEAR ===")
    print(df["year"].value_counts().sort_index())

    print()
    print("=== 2022–2024 ===")

    target = df[
        df["start_time"].between(
            "2022-01-01",
            "2024-12-31 23:59:59",
        )
    ].copy()

    print("Total:", len(target))

    print()
    print("By type:")
    print(target["type"].value_counts())

    print()
    print("By phase:")
    print(target["phase"].value_counts())

    print()
    print("=== SAMPLE TARGET CONTESTS ===")
    print(
        target[
            [
                "id",
                "name",
                "type",
                "phase",
                "start_time",
                "durationSeconds",
            ]
        ]
        .sort_values("start_time")
        .head(20)
        .to_string(index=False)
    )

    print()
    print("=== LAST TARGET CONTESTS ===")
    print(
        target[
            [
                "id",
                "name",
                "type",
                "phase",
                "start_time",
                "durationSeconds",
            ]
        ]
        .sort_values("start_time")
        .tail(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()