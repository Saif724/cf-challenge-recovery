from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "first_rated_contest.parquet"
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_sample_200.parquet"
)

SEED = 20260923

SAMPLE_SIZES = {
    2022: 67,
    2023: 67,
    2024: 66,
}


def main():
    df = pd.read_parquet(INPUT)

    df["first_year"] = (
        df["first_contest_start_time"]
        .dt.year
    )

    cohort = df[
        df["first_year"].isin(SAMPLE_SIZES)
    ].copy()

    print("=" * 70)
    print("CODEFORCES CHALLENGE-RECOVERY — PILOT SAMPLING")
    print("=" * 70)

    print()
    print("=== SAMPLING FRAME ===")
    print(f"Eligible users: {len(cohort):,}")

    print()
    print("=== TARGET SAMPLE ===")
    for year, size in SAMPLE_SIZES.items():
        print(f"{year}: {size}")

    samples = []

    for year, size in SAMPLE_SIZES.items():
        year_users = cohort[
            cohort["first_year"] == year
        ].copy()

        if len(year_users) < size:
            raise ValueError(
                f"Not enough users for {year}: "
                f"{len(year_users)} available, "
                f"{size} required."
            )

        sampled = year_users.sample(
            n=size,
            random_state=SEED + year,
        )

        samples.append(sampled)

    pilot = pd.concat(
        samples,
        ignore_index=True,
    )

    pilot = pilot.sort_values(
        ["first_year", "handle"]
    ).reset_index(drop=True)

    if len(pilot) != 200:
        raise ValueError(
            f"Expected 200 users, got {len(pilot)}."
        )

    if pilot["handle"].duplicated().any():
        raise ValueError(
            "Duplicate handles found in pilot sample."
        )

    print()
    print("=== FINAL SAMPLE ===")
    print(f"Users: {len(pilot):,}")

    print()
    print("=== SAMPLE BY YEAR ===")
    print(
        pilot["first_year"]
        .value_counts()
        .sort_index()
    )

    print()
    print("=== FIRST CONTEST TYPE ===")

    contest_file = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "contest_list.json"
    )

    import json

    with contest_file.open(
        "r",
        encoding="utf-8",
    ) as f:
        contests = pd.DataFrame(json.load(f))

    contest_types = contests[
        ["id", "type", "name"]
    ].rename(
        columns={
            "id": "first_contest_id",
            "type": "contest_type",
            "name": "first_contest_name_lookup",
        }
    )

    pilot_with_type = pilot.merge(
        contest_types,
        on="first_contest_id",
        how="left",
    )

    print(
        pilot_with_type["contest_type"]
        .value_counts(dropna=False)
    )

    print()
    print("=== YEAR × CONTEST TYPE ===")
    print(
        pd.crosstab(
            pilot_with_type["first_year"],
            pilot_with_type["contest_type"],
            margins=True,
        )
    )

    print()
    print("=== SAMPLE PREVIEW ===")
    print(
        pilot[
            [
                "handle",
                "first_contest_id",
                "first_contest_name",
                "first_contest_start_time",
                "initial_rating",
                "first_year",
            ]
        ].head(20).to_string(index=False)
    )

    pilot.to_parquet(
        OUTPUT,
        index=False,
    )

    print()
    print("=== SAVED ===")
    print(OUTPUT)


if __name__ == "__main__":
    main()