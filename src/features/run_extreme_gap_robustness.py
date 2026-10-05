import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


INPUT = "data/processed/pilot_monthly_development_panel.parquet"


def prepare_data(df):
    df = df.copy()

    df = df[
        df["rating_at_month_end"].notna()
        & (df["accepted_rated_count"] >= 1)
        & df["rating_gain_3m"].notna()
        & df["difficulty_gap_p75"].notna()
        & df["rating_before_month"].notna()
    ].copy()

    df["month_str"] = pd.to_datetime(
        df["month"]
    ).dt.strftime("%Y-%m")

    df["log_practice_volume"] = np.log1p(
        df["accepted_rated_count"]
    )

    return df


def run_model(df, exposure="difficulty_gap_p75"):

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
    ci = model.conf_int().loc[exposure]

    return {
        "N": int(model.nobs),
        "Users": df["handle"].nunique(),
        "Coef": coef,
        "SE": model.bse[exposure],
        "p": model.pvalues[exposure],
        "CI_low": ci[0],
        "CI_high": ci[1],
        "Effect_100pt": coef * 100,
    }


def main():

    print("=" * 90)
    print("EXTREME DIFFICULTY-GAP ROBUSTNESS")
    print("=" * 90)

    df = pd.read_parquet(INPUT)
    data = prepare_data(df)

    results = []

    # ---------------------------------------------------------------
    # 1. Baseline
    # ---------------------------------------------------------------

    results.append({
        "Specification": "Baseline",
        **run_model(data),
    })

    # ---------------------------------------------------------------
    # 2. Remove |gap| >= 750
    # ---------------------------------------------------------------

    subset = data[
        data["difficulty_gap_p75"].abs() < 750
    ].copy()

    results.append({
        "Specification": "|gap| < 750",
        **run_model(subset),
    })

    # ---------------------------------------------------------------
    # 3. Remove |gap| >= 1000
    # ---------------------------------------------------------------

    subset = data[
        data["difficulty_gap_p75"].abs() < 1000
    ].copy()

    results.append({
        "Specification": "|gap| < 1000",
        **run_model(subset),
    })

    # ---------------------------------------------------------------
    # 4. Winsorize at 1st / 99th percentiles
    # ---------------------------------------------------------------

    winsor = data.copy()

    low = winsor["difficulty_gap_p75"].quantile(0.01)
    high = winsor["difficulty_gap_p75"].quantile(0.99)

    winsor["gap_winsor"] = winsor[
        "difficulty_gap_p75"
    ].clip(
        lower=low,
        upper=high,
    )

    results.append({
        "Specification": "Winsorized 1%-99%",
        **run_model(
            winsor,
            exposure="gap_winsor",
        ),
    })

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------

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
    print("EXPOSURE RANGE")
    print("=" * 90)

    print(
        f"Original min : "
        f"{data['difficulty_gap_p75'].min():.2f}"
    )

    print(
        f"Original max : "
        f"{data['difficulty_gap_p75'].max():.2f}"
    )

    print(
        f"1% cutoff    : "
        f"{low:.2f}"
    )

    print(
        f"99% cutoff   : "
        f"{high:.2f}"
    )


if __name__ == "__main__":
    main()