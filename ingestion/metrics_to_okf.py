import pandas as pd
from pathlib import Path

def metrics_to_okf(csv_file, output_dir):

    df = pd.read_csv(csv_file)

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file_name = Path(csv_file).stem

    md_file = output_dir / f"{file_name}.md"

    content = f"""---
type: financial_metric
company: TCS
source: {Path(csv_file).name}
---

"""

    content += df.to_markdown(index=False)

    with open(md_file, "w", encoding="utf-8") as f:
        f.write(content)