"""
fix_figure_layout.py
────────────────────
Merges Quarto/pandoc "figure layout" tables in a .docx file.

Quarto renders layout-ncol: 2 with 4 images as:
  Table-1 (2-col) : row [img_a | img_b]
  <w:p framePr/>  : zero-size separator paragraph (page-break point!)
  Table-2 (2-col) : row [img_c | img_d]

This script turns that into one unified 2×2 table, eliminating the break.

Usage
─────
  python fix_figure_layout.py input.docx              # overwrites in place
  python fix_figure_layout.py input.docx output.docx  # saves to new file
"""

import copy
import sys

from docx import Document
from docx.oxml.ns import qn

# ── helpers ──────────────────────────────────────────────────────────────────

def _outer_col_count(tbl_el):
    """Number of columns in the outermost <w:tblGrid> of a table element."""
    tbl_grid = tbl_el.find(qn("w:tblGrid"))
    if tbl_grid is None:
        return 0
    return len(tbl_grid.findall(qn("w:gridCol")))


def is_layout_table(el, ncols=2):
    """True if el is a <w:tbl> whose outermost tblGrid has exactly `ncols` columns."""
    return el.tag == qn("w:tbl") and _outer_col_count(el) == ncols


def is_framePr_para(el):
    """True if el is a <w:p> that contains a <w:framePr> (Quarto layout separator)."""
    if el.tag != qn("w:p"):
        return False
    pPr = el.find(qn("w:pPr"))
    return pPr is not None and pPr.find(qn("w:framePr")) is not None


# ── main merge logic ──────────────────────────────────────────────────────────

def merge_layout_tables(docx_path: str, out_path: str | None = None) -> int:
    """
    Scans the document body for the pattern:
        layout-table  →  framePr-paragraph  →  layout-table
    and merges each such triple into a single table.

    Returns the number of merges performed.
    """
    doc = Document(docx_path)
    body = doc.element.body
    merges = 0

    # Repeat until no more pairs are found (handles >2 rows if needed)
    changed = True
    while changed:
        changed = False
        children = list(body)

        for i in range(len(children) - 2):
            t1, sep, t2 = children[i], children[i + 1], children[i + 2]

            if is_layout_table(t1) and is_framePr_para(sep) and is_layout_table(t2):
                # Move every <w:tr> from t2 into t1
                for tr in t2.findall(qn("w:tr")):
                    t1.append(copy.deepcopy(tr))

                # Remove the separator paragraph and the now-empty second table
                body.remove(sep)
                body.remove(t2)

                merges += 1
                changed = True
                break  # restart scan after modifying the list

    doc.save(out_path or docx_path)
    return merges


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_figure_layout.py input.docx [output.docx]")
        sys.exit(1)

    path_in  = sys.argv[1]
    path_out = sys.argv[2] if len(sys.argv) > 2 else None

    n = merge_layout_tables(path_in, path_out)
    target = path_out or path_in
    print(f"Done — {n} table merge(s) applied → {target}")
