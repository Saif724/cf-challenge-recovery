import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


INPUT = "data/processed/pilot_monthly_development_panel.parquet"


def prepare_data(df, min_practice=1):
    df = df.copy()

    df = df[
        df["rating_at_month_end"].notna()
        & (df["accepted_rated_count"] >= min_practice)
        & df["rating_gain_3m"].notna()
        & df["difficulty_gap_p75"].notna()
        & df["rating_before_month"].notna()
    ].copy()

    df["month_str"] = pd.to_datetime(df["month"]).dt.strftime("%Y-%m")

    df["log_practice_volume"] = np.log1p(
        df["accepted_rated_count"]
    )

    return df


def fit_model(df, formula, exposure):
    model = smf.ols(
        formula=formula,
        data=df,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["handle"]},
    )

    return {
        "N": int(model.nobs),
        "Users": df["handle"].nunique(),
        "Coef": model.params.get(exposure, np.nan),
        "SE": model.bse.get(exposure, np.nan),
        "p": model.pvalues.get(exposure, np.nan),
        "CI_low": model.conf_int().loc[exposure, 0],
        "CI_high": model.conf_int().loc[exposure, 1],
        "R2": model.rsquared,
    }


def main():

    print("=" * 90)
    print("PILOT FIXED-EFFECTS MODELS")
    print("=" * 90)

    df = pd.read_parquet(INPUT)

    primary = prepare_data(df, min_practice=1)

    print()
    print("PRIMARY SAMPLE")
    print("-" * 90)
    print(f"Observations : {len(primary)}")
    print(f"Users        : {primary['handle'].nunique()}")

    results = []

    # ---------------------------------------------------------------
    # Model 1: User FE + month FE
    # ---------------------------------------------------------------

    results.append(
        {
            "Model": "M1: User + Month FE",
            **fit_model(
                primary,
                """
                rating_gain_3m
                ~ difficulty_gap_p75
                + C(handle)
                + C(month_str)
                """,
                "difficulty_gap_p75",
            ),
        }
    )

    # ---------------------------------------------------------------
    # Model 2: + baseline rating
    # ---------------------------------------------------------------

    results.append(
        {
            "Model": "M2: + Baseline Rating",
            **fit_model(
                primary,
                """
                rating_gain_3m
                ~ difficulty_gap_p75
                + rating_before_month
                + C(handle)
                + C(month_str)
                """,
                "difficulty_gap_p75",
            ),
        }
    )

    # ---------------------------------------------------------------
    # Model 3: + practice volume
    # ---------------------------------------------------------------

    results.append(
        {
            "Model": "M3: + Practice Volume",
            **fit_model(
                primary,
                """
                rating_gain_3m
                ~ difficulty_gap_p75
                + rating_before_month
                + log_practice_volume
                + C(handle)
                + C(month_str)
                """,
                "difficulty_gap_p75",
            ),
        }
    )

    # ---------------------------------------------------------------
    # Model 4: explicit within-user centered exposure
    # ---------------------------------------------------------------

    centered = primary.copy()

    centered["gap_within"] = (
        centered["difficulty_gap_p75"]
        - centered.groupby("handle")["difficulty_gap_p75"]
        .transform("mean")
    )

    results.append(
        {
            "Model": "M4: Within-User Centered",
            **fit_model(
                centered,
                """
                rating_gain_3m
                ~ gap_within
                + rating_before_month
                + log_practice_volume
                + C(month_str)
                """,
                "gap_within",
            ),
        }
    )

    # ---------------------------------------------------------------
    # Practice-volume sensitivity
    # ---------------------------------------------------------------

    for threshold in [3, 5, 10]:

        subset = prepare_data(
            df,
            min_practice=threshold,
        )

        if subset["handle"].nunique() < 10:
            continue

        result = fit_model(
            subset,
            """
            rating_gain_3m
            ~ difficulty_gap_p75
            + rating_before_month
            + log_practice_volume
            + C(handle)
            + C(month_str)
            """,
            "difficulty_gap_p75",
        )

        results.append(
            {
                "Model": f"M3: n >= {threshold}",
                **result,
            }
        )

    # ---------------------------------------------------------------
    # Print compact table
    # ---------------------------------------------------------------

    result_df = pd.DataFrame(results)

    print()
    print("=" * 90)
    print("COMPACT MODEL RESULTS")
    print("=" * 90)

    print(
        result_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print()
    print("=" * 90)
    print("INTERPRETATION HELPER")
    print("=" * 90)

    print(
        "Coefficient = expected change in rating_gain_3m "
        "for a +1 rating-point increase in difficulty_gap_p75."
    )

    print(
        "Multiply Coef by 100 to interpret a +100-point "
        "increase in difficulty gap."
    )


if __name__ == "__main__":
    main()