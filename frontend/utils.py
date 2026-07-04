import pandas as pd


def load_financials():

    income = pd.read_csv(
        "data/financial_metrics/income_statement.csv"
    )

    balance = pd.read_csv(
        "data/financial_metrics/balance_sheet.csv"
    )

    cashflow = pd.read_csv(
        "data/financial_metrics/cash_flow.csv"
    )

    return (
        income,
        balance,
        cashflow
    )


def get_metric(
    df,
    metric_name
):

    row = df[
        df["Unnamed: 0"]
        == metric_name
    ]

    if row.empty:
        return None

    return row.iloc[0]