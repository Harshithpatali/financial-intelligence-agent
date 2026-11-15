import fitz
from pathlib import Path
def pdf_to_okf(pdf_path,output_path,doc_type):
 """Convert PDF to Markdown and preserve page boundaries for citations."""
 pdf_path,output_path=Path(pdf_path),Path(output_path); pages=[]
 with fitz.open(pdf_path) as pdf:
  for number,page in enumerate(pdf,start=1):pages.append(f"<!-- PAGE: {number} -->\n\n{page.get_text('text').strip()}")
 metadata=f"---\ntype: {doc_type}\ncompany: TCS\nsource: {pdf_path.name}\n---\n\n"
 output_path.parent.mkdir(parents=True,exist_ok=True); output_path.write_text(metadata+"\n\n".join(pages),encoding="utf-8")
