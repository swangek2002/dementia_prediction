"""
Post-process PROGRESS_REPORT_EN.tex to:
  1. Add vertical lines (|) between columns in all longtable.
  2. Replace booktabs rules (\toprule/\midrule/\bottomrule) with \hline.
  3. Add \hline after every row terminator (\\) inside longtable body.
  4. Auto-shrink wide tables (>6 cols) to \footnotesize or \scriptsize.
  5. Enable line-breaking in lstlisting blocks (breaklines=true) and use smaller font.
"""
import re
from pathlib import Path

TEX = Path('/Data0/swangek_data/991/CPRD/PROGRESS_REPORT_EN.tex')
text = TEX.read_text()

# ---------- 1. listings global settings -----------
listings_preamble = r"""
% --- custom lstset: enable line breaks and small font for code blocks ---
\lstset{
    basicstyle=\ttfamily\small,
    breaklines=true,
    breakatwhitespace=false,
    breakindent=0pt,
    columns=fullflexible,
    keepspaces=true,
    xleftmargin=2pt,
    xrightmargin=2pt,
    frame=single,
    framesep=4pt,
    rulesep=0pt
}
% -----------------------------------------------------------------------
"""
# Insert just after the existing \lstset lines
text = text.replace(
    "\\lstset{defaultdialect=[x86masm]Assembler}",
    "\\lstset{defaultdialect=[x86masm]Assembler}\n" + listings_preamble
)

# ---------- 2. Rewrite every longtable to use vertical lines + \hline ----------

def fix_table(m):
    """Take one \begin{longtable}...\end{longtable} block and reformat."""
    body = m.group(0)

    # Extract column specifier between '{@{}' and '@{}}'
    spec_match = re.search(r'\\begin\{longtable\}\[\]\{@\{\}(.*?)@\{\}\}', body, re.DOTALL)
    if not spec_match:
        return body
    spec = spec_match.group(1).strip()

    # Walk the spec, splitting on top-level column boundaries (depth 0 after closing brace).
    # Pandoc produces specifiers like: >{\raggedright\arraybackslash}p{(\linewidth - 4\tabcolsep) * \real{0.3333}}
    # which contain nested braces. Simple regex misparses.
    cols = []
    depth = 0
    cur = ''
    for ch in spec:
        cur += ch
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            # A column-spec ends when we return to depth=0 AFTER a 'p{...}' block.
            if depth == 0 and re.search(r'p\{[^{}]*\}$|p\{.*\}$', cur):
                cols.append(cur.strip())
                cur = ''
        elif ch in 'lcr' and depth == 0 and cur.strip() in ('l', 'c', 'r'):
            cols.append(cur.strip())
            cur = ''
    if cur.strip():
        cols.append(cur.strip())

    n_cols = len(cols)

    # Build new column spec with vertical lines: |col|col|...|col|
    new_spec = "|" + "|".join(cols) + "|"

    # Choose font size based on column count
    if n_cols > 8:
        size = "\\scriptsize"
    elif n_cols > 5:
        size = "\\footnotesize"
    else:
        size = ""

    # Replace the column spec inside the block (use lambda to avoid backslash escape issues)
    replacement = "\\begin{longtable}[]{" + new_spec + "}"
    body = re.sub(
        r'\\begin\{longtable\}\[\]\{@\{\}.*?@\{\}\}',
        lambda _: replacement,
        body, count=1, flags=re.DOTALL
    )

    # Replace booktabs rules with \hline
    body = re.sub(r'\\toprule\\noalign\{\}', r'\\hline', body)
    body = re.sub(r'\\midrule\\noalign\{\}', r'\\hline', body)
    body = re.sub(r'\\bottomrule\\noalign\{\}', r'\\hline', body)
    body = re.sub(r'\\toprule', r'\\hline', body)
    body = re.sub(r'\\midrule', r'\\hline', body)
    body = re.sub(r'\\bottomrule', r'\\hline', body)

    # Add \hline after every row terminator '\\' that is at end of line.
    # Avoid touching '\\' immediately followed by \endhead / \endlastfoot or already
    # followed by \hline. Use a finite-state walk.
    out_lines = []
    inside = False
    just_after_endhead = False
    for line in body.split('\n'):
        out_lines.append(line)
        if '\\begin{longtable}' in line:
            inside = True
            continue
        if '\\end{longtable}' in line:
            inside = False
            continue
        if not inside:
            continue
        # If this line ends with '\\' (row terminator), add \hline next
        stripped = line.rstrip()
        if stripped.endswith('\\\\') and not stripped.endswith('\\\\hline'):
            # Make sure next line isn't already \hline
            out_lines.append('\\hline')

    body = '\n'.join(out_lines)

    # Prepend font-size adjustment if needed (wrap in group)
    if size:
        body = '{' + size + '\n' + body + '\n}'

    return body

text = re.sub(
    r'\\begin\{longtable\}.*?\\end\{longtable\}',
    fix_table,
    text,
    flags=re.DOTALL
)

TEX.write_text(text)
print(f"Fixed {TEX}")
# Quick stats
n_tab = text.count(r'\begin{longtable}')
n_lst = text.count(r'\begin{lstlisting}')
print(f"  {n_tab} longtables reformatted, {n_lst} lstlistings (breaklines enabled).")
