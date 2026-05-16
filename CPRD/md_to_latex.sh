#!/bin/bash
# Convert PROGRESS_REPORT_EN.md to LaTeX + bundle for Overleaf upload.
# Usage: bash md_to_latex.sh

set -e
cd /Data0/swangek_data/991/CPRD

# 1. MD → LaTeX (requires conda env 'survivehr' with pandoc installed)
/home/swangek/miniconda3/bin/conda run -n survivehr pandoc \
    PROGRESS_REPORT_EN.md \
    -o PROGRESS_REPORT_EN.tex \
    --standalone \
    --syntax-highlighting=idiomatic \
    --resource-path=. \
    -V geometry:margin=1in \
    -V fontsize=11pt

# 2. Post-process: add table borders + enable lstlisting line-breaks
/home/swangek/miniconda3/envs/survivehr/bin/python fix_tex.py

# 3. Bundle into zip for Overleaf
rm -f progress_report_overleaf.zip
zip -q -r progress_report_overleaf.zip \
    PROGRESS_REPORT_EN.tex \
    figs/fig_*.png

echo "Done. Upload progress_report_overleaf.zip to Overleaf."
ls -la PROGRESS_REPORT_EN.tex progress_report_overleaf.zip
