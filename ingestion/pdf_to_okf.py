import fitz
from pathlib import Path

def pdf_to_okf(
        pdf_path,
        output_path,
        doc_type
):

    pdf = fitz.open(pdf_path)

    text = ""

    for page in pdf:
        text += page.get_text()

    metadata = f"""---
type: {doc_type}
company: TCS
source: {Path(pdf_path).name}
---

"""

    content = metadata + text

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(content)