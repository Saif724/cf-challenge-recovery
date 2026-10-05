from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_contest_problem_events.parquet"
)


def main():
    events = pd.read_parquet(EVENT_FILE)

    required = {
        "strict_upsolve",
        "solve_delay_hours",
    }

    missing = required - set(events.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    delay = events["solve_delay_hours"]

    events["upsolve_24h"] = (
        events["strict_upsolve"]
        & delay.notna()
        & (delay <= 24)
    )

    events["upsolve_7d"] = (
        events["strict_upsolve"]
        & delay.notna()
        & (delay <= 24 * 7)
    )

    events["upsolve_30d"] = (
        events["strict_upsolve"]
        & delay.notna()
        & (delay <= 24 * 30)
    )

    output = EVENT_FILE

    events.to_parquet(output, index=False)

    print("=" * 70)
    print("UPSOLVE TIME-WINDOW VARIABLES")
    print("=" * 70)

    strict_count = int(events["strict_upsolve"].sum())

    print()
    print(f"Total events:       {len(events):,}")
    print(f"Strict upsolves:    {strict_count:,}")
    print(f"Upsolve <= 24h:     {events['upsolve_24h'].sum():,}")
    print(f"Upsolve <= 7d:      {events['upsolve_7d'].sum():,}")
    print(f"Upsolve <= 30d:     {events['upsolve_30d'].sum():,}")

    if strict_count:
        print()
        print("Among strict upsolves:")

        for column in [
            "upsolve_24h",
            "upsolve_7d",
            "upsolve_30d",
        ]:
            count = int(events.loc[
                events["strict_upsolve"],
                column,
            ].sum())

            percentage = 100 * count / strict_count

            print(
                f"{column:15s}: "
                f"{count:4d} / {strict_count:4d} "
                f"({percentage:6.2f}%)"
            )

    print()
    print("Saved:", output)


if __name__ == "__main__":
    main()