import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


INPUT = "data/processed/pilot_monthly_development_panel.parquet"


def prepare_data(df, exposure):
    df = df.copy()

    df = df[
        df["rating_at_month_end"].notna()
        & (df["accepted_rated_count"] >= 1)
        & df["rating_gain_3m"].notna()
        & df[exposure].notna()
        & df["rating_before_month"].notna()
    ].copy()

    df["month_str"] = pd.to_datetime(df["month"]).dt.strftime("%Y-%m")
    df["log_practice_volume"] = np.log1p(df["accepted_rated_count"])

    return df


def run_model(df, exposure):

    formula = f"""
        rating_gain_3m
        ~ {exposure}
        + rating_before_month
        + log_practice_volume
        + C(handle)
        + C(month_str)
    """

    model = smf.ols(
        formula=formula,
        data=df,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["handle"]},
    )

    coef = model.params[exposure]

    return {
        "Exposure": exposure,
        "N": int(model.nobs),
        "Users": df["handle"].nunique(),
        "Coef": coef,
        "SE": model.bse[exposure],
        "p": model.pvalues[exposure],
        "CI_low": model.conf_int().loc[exposure, 0],
        "CI_high": model.conf_int().loc[exposure, 1],
        "R2": model.rsquared,
        "Effect_100pt": coef * 100,
    }


def main():

    print("=" * 90)
    print("EXPOSURE ROBUSTNESS — P50 vs P75 vs P90")
    print("=" * 90)

    df = pd.read_parquet(INPUT)

    exposures = [
        "difficulty_gap_p50",
        "difficulty_gap_p75",
        "difficulty_gap_p90",
    ]

    results = []

    for exposure in exposures:

        data = prepare_data(df, exposure)

        print(
            f"{exposure}: "
            f"N={len(data)}, "
            f"Users={data['handle'].nunique()}"
        )

        results.append(
            run_model(data, exposure)
        )

    result_df = pd.DataFrame(results)

    print()
    print("=" * 90)
    print("RESULTS")
    print("=" * 90)

    print(
        result_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print()
    print("=" * 90)
    print("INTERPRETATION")
    print("=" * 90)

    print(
        "Effect_100pt = estimated rating-gain difference "
        "associated with a 100-point larger difficulty gap."
    )


if __name__ == "__main__":
    main()