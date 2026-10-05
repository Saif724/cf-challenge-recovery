import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


PANEL_INPUT = "data/processed/pilot_monthly_development_panel.parquet"
PRACTICE_INPUT = "data/processed/pilot_monthly_practice_features.parquet"


def build_broad_dataset():

    panel = pd.read_parquet(PANEL_INPUT)

    practice = pd.read_parquet(PRACTICE_INPUT)

    # Keep only broad practice definition
    broad = practice[
        practice["practice_definition"] == "broad"
    ].copy()

    # Restrict to analytical cohort already used by the panel
    cohort = set(panel["handle"].unique())

    broad = broad[
        broad["handle"].isin(cohort)
    ].copy()

    # Rename practice variables so they cannot collide
    broad = broad.rename(
        columns={
            "accepted_rated_count": "broad_accepted_rated_count",
            "practice_p50": "broad_practice_p50",
            "practice_p75": "broad_practice_p75",
            "practice_p90": "broad_practice_p90",
            "difficulty_gap_p50": "broad_difficulty_gap_p50",
            "difficulty_gap_p75": "broad_difficulty_gap_p75",
            "difficulty_gap_p90": "broad_difficulty_gap_p90",
            "rating_before_month": "broad_rating_before_month",
        }
    )

    broad = broad[
        [
            "handle",
            "month",
            "broad_accepted_rated_count",
            "broad_practice_p50",
            "broad_practice_p75",
            "broad_practice_p90",
            "broad_difficulty_gap_p50",
            "broad_difficulty_gap_p75",
            "broad_difficulty_gap_p90",
            "broad_rating_before_month",
        ]
    ]

    # Merge broad practice information onto the already validated panel
    data = panel.merge(
        broad,
        on=["handle", "month"],
        how="left",
        validate="one_to_one",
    )

    # We only analyze months with broad practice
    data = data[
        data["broad_difficulty_gap_p75"].notna()
    ].copy()

    # Same minimum practice requirement as the primary analysis
    data = data[
        data["broad_accepted_rated_count"] >= 1
    ].copy()

    # Same outcome and baseline requirements
    data = data[
        data["rating_at_month_end"].notna()
        & data["rating_gain_3m"].notna()
        & data["broad_rating_before_month"].notna()
    ].copy()

    data["month_str"] = pd.to_datetime(
        data["month"]
    ).dt.strftime("%Y-%m")

    data["log_broad_practice_volume"] = np.log1p(
        data["broad_accepted_rated_count"]
    )

    return data


def run_model(data, exposure):

    formula = f"""
        rating_gain_3m
        ~ {exposure}
        + broad_rating_before_month
        + log_broad_practice_volume
        + C(handle)
        + C(month_str)
    """

    model = smf.ols(
        formula=formula,
        data=data,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["handle"]},
    )

    coef = model.params[exposure]
    ci = model.conf_int().loc[exposure]

    return {
        "N": int(model.nobs),
        "Users": data["handle"].nunique(),
        "Coef": coef,
        "SE": model.bse[exposure],
        "p": model.pvalues[exposure],
        "CI_low": ci[0],
        "CI_high": ci[1],
        "Effect_100pt": coef * 100,
    }


def main():

    print("=" * 90)
    print("CLEAN vs BROAD PRACTICE DEFINITION")
    print("=" * 90)

    # ---------------------------------------------------------------
    # Primary clean model
    # ---------------------------------------------------------------

    panel = pd.read_parquet(PANEL_INPUT)

    clean = panel[
        panel["difficulty_gap_p75"].notna()
        & (panel["accepted_rated_count"] >= 1)
        & panel["rating_at_month_end"].notna()
        & panel["rating_gain_3m"].notna()
        & panel["rating_before_month"].notna()
    ].copy()

    clean["month_str"] = pd.to_datetime(
        clean["month"]
    ).dt.strftime("%Y-%m")

    clean["log_practice_volume"] = np.log1p(
        clean["accepted_rated_count"]
    )

    clean_formula = """
        rating_gain_3m
        ~ difficulty_gap_p75
        + rating_before_month
        + log_practice_volume
        + C(handle)
        + C(month_str)
    """

    clean_model = smf.ols(
        formula=clean_formula,
        data=clean,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": clean["handle"]},
    )

    clean_coef = clean_model.params[
        "difficulty_gap_p75"
    ]

    clean_ci = clean_model.conf_int().loc[
        "difficulty_gap_p75"
    ]

    clean_result = {
        "Definition": "Clean",
        "N": int(clean_model.nobs),
        "Users": clean["handle"].nunique(),
        "Coef": clean_coef,
        "SE": clean_model.bse["difficulty_gap_p75"],
        "p": clean_model.pvalues["difficulty_gap_p75"],
        "CI_low": clean_ci[0],
        "CI_high": clean_ci[1],
        "Effect_100pt": clean_coef * 100,
    }

    # ---------------------------------------------------------------
    # Broad model
    # ---------------------------------------------------------------

    broad = build_broad_dataset()

    broad_result = {
        "Definition": "Broad",
        **run_model(
            broad,
            "broad_difficulty_gap_p75",
        ),
    }

    results = pd.DataFrame(
        [
            clean_result,
            broad_result,
        ]
    )

    print()
    print("=" * 90)
    print("RESULTS")
    print("=" * 90)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print()
    print("=" * 90)
    print("PRACTICE COVERAGE")
    print("=" * 90)

    print(
        f"Clean exposure months : "
        f"{clean['month_str'].nunique()} unique months"
    )

    print(
        f"Broad exposure rows   : "
        f"{len(broad)}"
    )

    print(
        f"Broad users            : "
        f"{broad['handle'].nunique()}"
    )


if __name__ == "__main__":
    main()