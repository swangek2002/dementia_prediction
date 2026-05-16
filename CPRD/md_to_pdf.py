"""Convert PROGRESS_REPORT_EN.md to PDF using markdown-pdf."""
from pathlib import Path
from markdown_pdf import MarkdownPdf, Section

REPORT_DIR = Path('/Data0/swangek_data/991/CPRD')
MD = REPORT_DIR / 'PROGRESS_REPORT_EN.md'
PDF = REPORT_DIR / 'PROGRESS_REPORT_EN.pdf'

text = MD.read_text()

pdf = MarkdownPdf(toc_level=2, optimize=True)
# Pass the markdown root so relative image paths (figs/*.png) resolve correctly
pdf.add_section(Section(text, root=str(REPORT_DIR)))

# Title metadata
pdf.meta["title"] = "Dementia Risk Prediction — Progress Report"
pdf.meta["author"] = "swangek2002"

pdf.save(str(PDF))
print(f"Saved {PDF} ({PDF.stat().st_size / 1024:.1f} KB)")
