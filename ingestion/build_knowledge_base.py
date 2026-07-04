from pathlib import Path

from pdf_to_okf import pdf_to_okf
from metrics_to_okf import metrics_to_okf
from news_to_okf import convert_news
from company_profile_to_okf import copy_company_profile


# ==========================
# Annual Reports
# ==========================
from pathlib import Path

folders = [
    "knowledge/annual_reports",
    "knowledge/quarterly_reports",
    "knowledge/financial_metrics",
    "knowledge/company_profile",
    "knowledge/news"
]

for folder in folders:
    Path(folder).mkdir(
        parents=True,
        exist_ok=True
    )
for pdf in Path("data/annual_reports").glob("*.pdf"):

    output = (
        Path("knowledge/annual_reports")
        / f"{pdf.stem}.md"
    )

    pdf_to_okf(
        pdf,
        output,
        "annual_report"
    )


# ==========================
# Quarterly Reports
# ==========================

for pdf in Path("data/quarterly_reports").glob("*.pdf"):

    output = (
        Path("knowledge/quarterly_reports")
        / f"{pdf.stem}.md"
    )

    pdf_to_okf(
        pdf,
        output,
        "quarterly_report"
    )


# ==========================
# Financial Metrics
# ==========================

for csv_file in Path(
        "data/financial_metrics"
).glob("*.csv"):

    metrics_to_okf(
        csv_file,
        "knowledge/financial_metrics"
    )


# ==========================
# News
# ==========================

convert_news(
    "data/news/tcs_news.csv",
    "knowledge/news"
)


# ==========================
# Company Profile
# ==========================

copy_company_profile()


print("\nKnowledge Base Created Successfully!")