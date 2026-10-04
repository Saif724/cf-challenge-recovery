from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "first_rated_contest.parquet"
)


def main():
    df = pd.read_parquet(INPUT)

    print("=" * 70)
    print("FIRST RATED CONTEST — POPULATION INSPECTION")
    print("=" * 70)

    print()
    print("=== BASIC ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique handles: {df['handle'].nunique():,}")

    print()
    print("=== DUPLICATES ===")
    print(
        "Duplicate handles:",
        df["handle"].duplicated().sum()
    )

    print(
        "Duplicate first contest records:",
        df.duplicated().sum()
    )

    print()
    print("=== INITIAL RATING ===")
    print(df["initial_rating"].describe())

    print()
    print("=== INITIAL RATING QUANTILES ===")
    print(
        df["initial_rating"].quantile(
            [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        )
    )

    print()
    print("=== FIRST RATED YEAR ===")
    print(
        df["first_contest_start_time"]
        .dt.year
        .value_counts()
        .sort_index()
    )

    print()
    print("=== FIRST RATED MONTH ===")
    print(
        df["first_contest_start_time"]
        .dt.to_period("M")
        .value_counts()
        .sort_index()
    )

    print()
    print("=== FIRST CONTESTS ===")
    print(
        df["first_contest_name"]
        .value_counts()
        .head(30)
    )

    print()
    print("=== FIRST CONTEST IDs ===")
    print(
        df["first_contest_id"]
        .value_counts()
        .head(30)
    )

    print()
    print("=== TARGET COHORT ===")

    cohort = df[
        df["first_contest_start_time"].dt.year.between(2022, 2024)
    ].copy()

    print()
    print("=== FIRST CONTEST TYPE ===")

    contest_file = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "contest_list.json"
    )

    with contest_file.open("r", encoding="utf-8") as f:
        contests = pd.DataFrame(__import__("json").load(f))

    contest_types = contests[
        ["id", "type", "phase", "name"]
    ].copy()

    contest_types = contest_types.rename(
        columns={
            "id": "first_contest_id",
            "type": "contest_type",
            "phase": "contest_phase",
            "name": "contest_name",
        }
    )

    cohort_with_type = cohort.merge(
        contest_types,
        on="first_contest_id",
        how="left",
    )

    print(
        cohort_with_type["contest_type"]
        .value_counts(dropna=False)
    )

    print()
    print("=== FIRST CONTEST TYPE × YEAR ===")

    print(
        pd.crosstab(
            cohort_with_type["first_contest_start_time"].dt.year,
            cohort_with_type["contest_type"],
            margins=True,
        )
    )

    print()
    print("=== FIRST CONTEST PHASE ===")

    print(
        cohort_with_type["contest_phase"]
        .value_counts(dropna=False)
    )

    print()
    print("=== FIRST CONTEST TYPE × PHASE ===")

    print(
        cohort_with_type.groupby(
            ["contest_type", "contest_phase"]
        ).size()
        .sort_values(ascending=False)
    )

    print(f"Users: {len(cohort):,}")

    print()
    print("=== TOP FIRST CONTESTS BY TYPE ===")

    contest_lookup = contests[
        ["id", "type", "phase", "name"]
    ].copy()

    contest_lookup = contest_lookup.rename(
        columns={
            "id": "first_contest_id",
            "type": "contest_type",
            "phase": "contest_phase",
            "name": "contest_name",
        }
    )

    contest_counts = (
        cohort_with_type
        .groupby(
            [
                "first_contest_id",
                "contest_type",
                "contest_phase",
                "contest_name",
            ]
        )
        .size()
        .reset_index(name="users")
        .sort_values("users", ascending=False)
    )

    print()
    print("--- ICPC ---")
    print(
        contest_counts[
            contest_counts["contest_type"] == "ICPC"
        ].head(20).to_string(index=False)
    )

    print()
    print("--- CF ---")
    print(
        contest_counts[
            contest_counts["contest_type"] == "CF"
        ].head(20).to_string(index=False)
    )

    print()
    print("=== TARGET INITIAL RATING ===")
    print(cohort["initial_rating"].describe())

    print()
    print("=== TARGET INITIAL RATING BANDS ===")

    bands = pd.cut(
        cohort["initial_rating"],
        bins=[
            float("-inf"),
            999,
            1199,
            1399,
            1599,
            1799,
            1999,
            float("inf"),
        ],
        labels=[
            "<1000",
            "1000–1199",
            "1200–1399",
            "1400–1599",
            "1600–1799",
            "1800–1999",
            "2000+",
        ],
    )

    print(
        bands.value_counts()
        .sort_index()
    )

    print()
    print("=== TARGET YEAR × INITIAL RATING BAND ===")

    target_year = (
        cohort["first_contest_start_time"]
        .dt.year
        .rename("first_year")
    )

    cross = pd.crosstab(
        target_year,
        bands,
    )

    print(cross)


if __name__ == "__main__":
    main()