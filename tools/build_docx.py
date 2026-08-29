"""Assemble UAV_Swarm_Mosaicking_Technical_Documentation.docx.

    python tools/build_docx.py

Generates the figures first if they are missing, then writes the document to the project
root. Safe to re-run; it overwrites.
"""

from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from make_docx import Doc, OUT_DOCX, FIG, add_footer, title_page, contents  # noqa: E402
from docx_part1 import section1, section2  # noqa: E402
from docx_part2 import section3, section4  # noqa: E402
from docx_part3 import section5, section6, appendices  # noqa: E402

NEEDED = ("fig_frames.png", "fig_raypl.png", "fig_parallax.png", "fig_flow.png",
          "fig_packet.png", "doc_video.png", "doc_tiles.png", "doc_hud.png",
          "compare.png", "seam_compare.png")


def ensure_figures():
    missing = [f for f in NEEDED if not os.path.exists(os.path.join(FIG, f))]
    if not missing:
        return
    print("missing figures:", ", ".join(missing))
    diagrams = [f for f in missing if f.startswith("fig_")]
    if diagrams:
        print("  regenerating diagrams...")
        subprocess.run([sys.executable, os.path.join(HERE, "make_doc_figures.py")],
                       cwd=ROOT, check=True)
    still = [f for f in NEEDED if not os.path.exists(os.path.join(FIG, f))]
    if still:
        print("  !! screenshots not present, those figures will be skipped:",
              ", ".join(still))


def main() -> int:
    ensure_figures()
    doc = Doc()
    add_footer(doc)

    title_page(doc)
    contents(doc)
    section1(doc)
    section2(doc)
    section3(doc)
    section4(doc)
    section5(doc)
    section6(doc)
    appendices(doc)

    doc.d.save(OUT_DOCX)
    size = os.path.getsize(OUT_DOCX) / 1024
    print(f"\nwrote {OUT_DOCX}")
    print(f"  {size:,.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
