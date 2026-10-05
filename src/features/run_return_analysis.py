import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


INPUT = "data/processed/pilot_monthly_development_panel.parquet"


def prepare_data(df):

    df = df.copy()

    # We need an exposure and baseline state.
    df = df[
        df["difficulty_gap_p75"].notna()
        & df["rating_before_month"].notna()
        & df["rating_at_month_end"].notna()
        & (df["accepted_rated_count"] >= 1)
    ].copy()

    df["month_str"] = pd.to_datetime(
        df["month"]
    ).dt.strftime("%Y-%m")

    df["log_practice_volume"] = np.log1p(
        df["accepted_rated_count"]
    )

    # rating_gain_3m is only non-null when a future
    # rated contest exists within the 3-month window.
    df["returned_within_3m"] = (
        df["rating_gain_3m"].notna()
    ).astype(int)

    return df


def main():

    print("=" * 90)
    print("RETURN-TO-CONTEST ANALYSIS")
    print("=" * 90)

    df = pd.read_parquet(INPUT)

    data = prepare_data(df)

    print()
    print("SAMPLE")
    print("-" * 90)
    print(f"Observations : {len(data)}")
    print(f"Users        : {data['handle'].nunique()}")

    print()
    print("RETURN DISTRIBUTION")
    print("-" * 90)

    counts = data["returned_within_3m"].value_counts()

    returned = int(counts.get(1, 0))
    not_returned = int(counts.get(0, 0))

    print(f"Returned     : {returned}")
    print(f"Did not return: {not_returned}")
    print(
        f"Return rate  : "
        f"{returned / len(data):.4f}"
    )

    # ---------------------------------------------------------------
    # Linear probability model
    # ---------------------------------------------------------------

    formula = """
        returned_within_3m
        ~ difficulty_gap_p75
        + rating_before_month
        + log_practice_volume
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

    coef = model.params["difficulty_gap_p75"]

    print()
    print("=" * 90)
    print("RETURN MODEL")
    print("=" * 90)

    print(f"N             : {int(model.nobs)}")
    print(f"Users         : {data['handle'].nunique()}")
    print(f"Coefficient   : {coef:.6f}")
    print(
        f"Std. Error    : "
        f"{model.bse['difficulty_gap_p75']:.6f}"
    )
    print(
        f"p-value       : "
        f"{model.pvalues['difficulty_gap_p75']:.6f}"
    )

    ci = model.conf_int().loc[
        "difficulty_gap_p75"
    ]

    print(
        f"95% CI        : "
        f"[{ci[0]:.6f}, {ci[1]:.6f}]"
    )

    print(
        f"Effect / 100 pts: "
        f"{coef * 100:.6f}"
    )

    print()
    print(
        "Interpretation: coefficient is the change in "
        "probability of returning within 3 months "
        "associated with a +1 rating-point increase "
        "in difficulty gap."
    )


if __name__ == "__main__":
    main()