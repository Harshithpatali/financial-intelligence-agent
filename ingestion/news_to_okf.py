import pandas as pd
from pathlib import Path

def convert_news(csv_file, output_dir):

    df = pd.read_csv(csv_file)

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for idx, row in df.iterrows():

        title = str(row["title"])

        md = f"""---
type: news
company: TCS
date: {row.get('published', '')}
source: Google News
---

# {title}

Link:
{row['link']}
"""

        filename = output_dir / f"news_{idx}.md"

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(md)