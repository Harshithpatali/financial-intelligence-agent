import pandas as pd

income = pd.read_csv(
    "data/financial_metrics/income_statement.csv"
)

print(income["Unnamed: 0"].tolist())