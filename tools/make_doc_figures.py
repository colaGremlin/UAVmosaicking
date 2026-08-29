"""Generate the vector-quality figures embedded in the technical documentation.

Run:  python tools/make_doc_figures.py
Writes PNGs into out/fig_*.png at 200 dpi, sized for a Word page (6.5 in text width).
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

INK = "#1A1F1E"
MUT = "#5C6461"
ACC = "#0B5D57"
WRN = "#A85B12"
STP = "#8E2128"
GRD = "#C9CCC7"
PAPER = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.edgecolor": GRD,
    "text.color": INK,
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
})

OUT = "out"
os.makedirs(OUT, exist_ok=True)


def _save(fig, name):
    p = f"{OUT}/{name}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("  wrote", p)


def _axes3(ax, ox, oy, axes, title, sub, colour=INK):
    """Draw a labelled 3-axis gnomon. axes = [(dx, dy, label, lx, ly, ha, va), ...]."""
    for dx, dy, lab, lx, ly, ha, va in axes:
        ax.add_patch(FancyArrowPatch((ox, oy), (ox + dx, oy + dy),
                                     arrowstyle="-|>", mutation_scale=10,
                                     lw=1.5, color=colour, zorder=3))
        ax.text(ox + lx, oy + ly, lab, ha=ha, va=va,
                fontsize=7.6, fontweight="bold", color=colour, zorder=4)
    ax.text(ox, -1.45, title, ha="center", va="top", fontsize=9.2, fontweight="bold")
    ax.text(ox, -1.80, sub, ha="center", va="top", fontsize=7.4, color=MUT)


# ---------------------------------------------------------------------------------
# Figure 1 -- coordinate frames and the two bridging matrices
# ---------------------------------------------------------------------------------
def fig_frames():
    fig, ax = plt.subplots(figsize=(6.5, 2.7))
    ax.set_xlim(-1.5, 14.6)
    ax.set_ylim(-3.1, 2.9)
    ax.axis("off")

    UP = [(0.95, 0, "X$_u$  right", 1.12, 0.0, "left", "center"),
          (0, 0.95, "Y$_u$  up", 0.0, 1.16, "center", "bottom"),
          (0.58, 0.58, "Z$_u$  fwd", 0.72, 0.66, "left", "bottom")]
    EN = [(0.95, 0, "E  east", 1.12, 0.0, "left", "center"),
          (0, 0.95, "U  up", 0.0, 1.16, "center", "bottom"),
          (0.58, 0.58, "N  north", 0.72, 0.66, "left", "bottom")]
    CV = [(0.95, 0, "X  right", 1.12, 0.0, "left", "center"),
          (0, -0.95, "Y  down", -0.10, -1.05, "right", "top"),
          (0.58, 0.58, "Z  optical", 0.72, 0.66, "left", "bottom")]

    _axes3(ax, 0.4, 0, UP, "Unity world", "left-handed", INK)
    _axes3(ax, 6.0, 0, EN, "Local ENU", "right-handed", ACC)
    _axes3(ax, 11.3, 0, CV, "CV camera", "right-handed", ACC)

    for x0, x1, lab, sub in ((2.5, 4.5, r"$\mathbf{S}$", "E=x$_u$,  N=z$_u$,  U=y$_u$"),
                             (8.1, 10.1, r"$\mathbf{F}$", "flip the camera Y axis")):
        ax.add_patch(FancyArrowPatch((x0, 2.05), (x1, 2.05), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.8, color=WRN))
        ax.text((x0 + x1) / 2, 2.28, lab, ha="center", fontsize=11.5,
                color=WRN, fontweight="bold")
        ax.text((x0 + x1) / 2, 1.72, sub, ha="center", va="top", fontsize=6.9, color=MUT)

    ax.text(6.6, -2.72,
            r"$\mathbf{R}_{E \leftarrow C} = \mathbf{S}\,\mathbf{R}_u\,\mathbf{F}$"
            "          det = (−1)(+1)(−1) = +1,  a proper rotation",
            ha="center", fontsize=9.6, color=INK)
    _save(fig, "fig_frames")


# ---------------------------------------------------------------------------------
# Figure 2 -- ray-plane intersection and the 3-tier ground plane cascade
# ---------------------------------------------------------------------------------
def fig_raypl():
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.set_xlim(-0.5, 12.6)
    ax.set_ylim(-1.4, 6.4)
    ax.axis("off")

    cam = (1.5, 5.2)

    ax.add_patch(Polygon([(-0.3, 0.55), (2.0, 0.95), (4.2, 0.35), (6.6, 1.12),
                          (9.0, 0.5), (12.4, 0.85), (12.4, -1.2), (-0.3, -1.2)],
                         closed=True, fc="#E8E4DA", ec="#B9B2A2", lw=1.0, zorder=1))
    ax.text(12.2, -0.75, "true terrain", ha="right", fontsize=7.5, color="#8A8272", style="italic")

    ax.plot([-0.3, 12.4], [0.85, 0.85], color=ACC, lw=1.9, zorder=2)
    ax.text(-0.25, 1.06, r"assumed ground plane   $z = z_{plane}$",
            fontsize=8, color=ACC, fontweight="bold")

    for tx in (3.9, 6.0, 8.0, 9.6):
        ax.plot([cam[0], tx], [cam[1] - 0.2, 0.85], color=MUT, lw=0.9, zorder=3)
    ax.plot([cam[0], 6.0], [cam[1] - 0.2, 0.85], color=STP, lw=1.8, zorder=4)

    ax.plot([3.9, 9.6], [0.85, 0.85], color=STP, lw=4.5, solid_capstyle="butt", zorder=4, alpha=.45)
    ax.annotate("", xy=(3.9, 0.28), xytext=(9.6, 0.28),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0))
    ax.text(6.75, 0.02, "ground footprint  →  4 corners  →  homography", ha="center", fontsize=8)

    ax.plot([cam[0], cam[0]], [cam[1] - 0.2, 0.85], color=MUT, lw=0.9, ls=":", zorder=3)
    ax.text(cam[0] - 0.14, 2.9, "nadir", rotation=90, ha="right", va="center",
            fontsize=7.5, color=MUT)
    ax.annotate("$\\theta$", xy=(2.02, 3.05), fontsize=12, color=WRN)

    ax.add_patch(FancyBboxPatch((cam[0] - 0.36, cam[1] - 0.2), 0.72, 0.4,
                                boxstyle="round,pad=0.05", fc=INK, ec="none", zorder=6))
    ax.text(cam[0] + 0.55, cam[1] + 0.02, r"camera  $\mathbf{C}$", ha="left", va="center",
            fontsize=9, fontweight="bold", zorder=7)

    ax.text(3.05, 4.25, r"$\mathbf{d}_w = \mathbf{R}_{E \leftarrow C}\,\mathbf{K}^{-1}[u,v,1]^T$",
            fontsize=9.5, color=STP)
    ax.text(3.05, 3.72, r"$\lambda = (z_{plane} - C_z)\,/\,d_{w,z}$", fontsize=9.5, color=STP)
    ax.text(3.05, 3.19, r"$\mathbf{G} = \mathbf{C} + \lambda\,\mathbf{d}_w$", fontsize=9.5, color=STP)

    ax.add_patch(FancyBboxPatch((8.35, 2.75), 4.0, 3.25, boxstyle="round,pad=0.14",
                                fc="#F4F6F5", ec=ACC, lw=1.1, zorder=6))
    ax.text(8.55, 5.72, "ground plane cascade", fontsize=8.2, fontweight="bold",
            color=ACC, zorder=7)
    ax.text(8.55, 5.44, "first rule that yields a valid range wins",
            fontsize=6.8, color=MUT, style="italic", zorder=7)
    for i, (t, d) in enumerate((("1   LRF slant range", r"$z=(\mathbf{C}+r\,\mathbf{d}_{bore})_z$"),
                                ("2   nadir AGL probe", r"$z=C_z-\mathrm{AGL}$"),
                                ("3   AOI default", r"$z=z_{default}$"))):
        y = 5.02 - i * 0.78
        ax.text(8.55, y, t, fontsize=7.9, fontweight="bold", zorder=7)
        ax.text(8.75, y - 0.32, d, fontsize=7.9, color=MUT, zorder=7)

    _save(fig, "fig_raypl")


# ---------------------------------------------------------------------------------
# Figure 3 -- parallax error from the flat-plane assumption
# ---------------------------------------------------------------------------------
def fig_parallax():
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    ax.set_xlim(-0.4, 10.6)
    ax.set_ylim(-0.8, 4.6)
    ax.axis("off")

    cam = (1.2, 4.1)
    ax.add_patch(FancyBboxPatch((cam[0] - 0.32, cam[1] - 0.18), 0.64, 0.36,
                                boxstyle="round,pad=0.04", fc=INK, ec="none", zorder=5))
    ax.text(cam[0], cam[1] + 0.38, "camera", ha="center", fontsize=8.5, fontweight="bold")

    ax.plot([-0.2, 10.4], [0.6, 0.6], color=ACC, lw=1.8)
    ax.text(-0.15, 0.78, "assumed plane", fontsize=8, color=ACC, fontweight="bold")

    bx, bw, bh = 6.1, 0.85, 1.25
    ax.add_patch(FancyBboxPatch((bx, 0.6), bw, bh, boxstyle="square,pad=0",
                                fc="#DDD8CC", ec="#9A927F", lw=1.1, zorder=3))
    ax.text(bx + bw / 2, 0.6 + bh + 0.16, "building, height $h$", ha="center", fontsize=8)

    top = (bx + bw / 2, 0.6 + bh)
    dx, dy = top[0] - cam[0], top[1] - cam[1]
    t = (0.6 - cam[1]) / dy
    land = (cam[0] + dx * t, 0.6)
    ax.plot([cam[0], land[0]], [cam[1], land[1]], color=STP, lw=1.4, zorder=4)
    ax.plot([top[0], land[0]], [0.6, 0.6], color=STP, lw=3.5, alpha=.45, zorder=4)

    ax.annotate("", xy=(top[0], 0.32), xytext=(land[0], 0.32),
                arrowprops=dict(arrowstyle="<->", color=STP, lw=1.1))
    ax.text((top[0] + land[0]) / 2, 0.06, r"displacement $\approx h\,\tan\theta$",
            ha="center", fontsize=9, color=STP, fontweight="bold")

    ax.plot([cam[0], cam[0]], [cam[1], 0.6], color=MUT, lw=0.9, ls=":")
    ax.annotate(r"$\theta$", xy=(1.62, 3.1), fontsize=11, color=WRN)

    ax.text(10.3, 3.9, "the roof is drawn where the ray meets the plane,\n"
                       "not where the roof actually is",
            ha="right", va="top", fontsize=8, color=MUT, style="italic")
    _save(fig, "fig_parallax")


# ---------------------------------------------------------------------------------
# Figure 4 -- dataflow and thread topology
# ---------------------------------------------------------------------------------
def fig_flow():
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")

    def box(x, y, w, h, title, lines, fc="#FFFFFF", ec=INK, tc=INK, lw=1.1, ts=8.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                                    fc=fc, ec=ec, lw=lw, zorder=3))
        ax.text(x + w / 2, y + h - 3.0, title, ha="center", va="center", fontsize=ts,
                fontweight="bold", color=tc, zorder=4)
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 6.6 - i * 3.1, ln, ha="center", va="center",
                    fontsize=6.6, color=MUT, zorder=4)

    def arrow(x0, y0, x1, y1, col=INK, lw=1.2):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=9, lw=lw, color=col, zorder=2))

    ax.text(11.0, 62.5, "AIRCRAFT", ha="center", fontsize=7.2,
            fontweight="bold", color=MUT)
    ax.text(51.0, 62.5, "GROUND STATION — one process, daemon threads", ha="center",
            fontsize=7.2, fontweight="bold", color=MUT)
    ax.text(89.0, 62.5, "OPERATOR", ha="center", fontsize=7.2, fontweight="bold", color=MUT)
    ax.plot([0.5, 99.5], [60.4, 60.4], color=GRD, lw=0.8)

    ROWS = (45.5, 33.0, 20.5, 8.0)
    for i, y in enumerate(ROWS):
        box(1.5, y, 19, 11, f"UAV {i}", ["EO capture + pose", f"UDP  :{5001 + i}"],
            fc="#F7F8F7")
        arrow(21.2, y + 5.5, 26.8, y + 5.5, col=MUT, lw=1.0)
        box(27, y, 20, 11, f"RxThread {i}", ["reassemble, decode", "1-deep mailbox"],
            fc="#F4F6F5", ec=ACC, tc=ACC)
        arrow(47.7, y + 5.5, 52.8, 31.0, col=MUT, lw=0.9)

    box(53, 22, 24, 19, "FusionEngine   10 Hz",
        ["snapshot 4 mailboxes", "drop stale > 0.5 s",
         "warp pool, 4 threads", "max-weight composite"],
        fc="#EAF1F0", ec=ACC, tc=ACC, lw=1.6, ts=7.6)

    box(79, 42, 20, 11, "Map layer  :8081", ["XYZ tiles + WMS", "→ MP map screen"],
        fc="#F4F6F5", ec=ACC, tc=ACC, ts=7.6)
    box(79, 27, 20, 11, "MJPEG  :8080", ["HTTP multipart", "→ MP HUD pane"], ts=7.6)
    box(79, 12, 20, 11, "FFmpeg → :5600", ["H.264 / MPEG-TS", "→ ffplay, VLC"], ts=7.6)

    arrow(77.7, 34.5, 78.6, 47.5, col=ACC, lw=1.4)
    arrow(77.7, 32.0, 78.6, 32.5, col=MUT, lw=1.1)
    arrow(77.7, 29.5, 78.6, 17.5, col=MUT, lw=1.1)

    ax.text(50, 3.6, "Every stage releases the GIL (cv2.imdecode, warpPerspective, NumPy), so the "
                     "threads run genuinely in parallel.",
            ha="center", fontsize=7.0, color=MUT, style="italic")
    ax.text(50, 0.8, "Mailboxes are 1-deep, not queues: under overload the newest frame wins and "
                     "the older one is discarded.",
            ha="center", fontsize=7.0, color=MUT, style="italic")
    _save(fig, "fig_flow")


# ---------------------------------------------------------------------------------
# Figure 5 -- packet layout
# ---------------------------------------------------------------------------------
def fig_packet():
    fig, ax = plt.subplots(figsize=(6.5, 2.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    segs = [(0, 26, "HEADER\n36 B", "#D8E4E2", ACC),
            (26, 46, "TELEMETRY\n64 B", "#F1E6D4", WRN),
            (46, 100, "JPEG SLICE\n≤ 1300 B", "#F0F0EC", MUT)]
    for x0, x1, lab, fc, ec in segs:
        ax.add_patch(FancyBboxPatch((x0 + .5, 9), x1 - x0 - 1, 12,
                                    boxstyle="round,pad=0.3", fc=fc, ec=ec, lw=1.3))
        ax.text((x0 + x1) / 2, 15, lab, ha="center", va="center",
                fontsize=8.4, fontweight="bold", color=INK)

    ax.annotate("", xy=(0.5, 24.5), xytext=(99.5, 24.5),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0))
    ax.text(50, 26, "one UDP datagram  ≤ 1400 B", ha="center", fontsize=8.4, fontweight="bold")

    ax.text(50, 4.5, "Telemetry repeats in EVERY fragment (~5 % overhead). Pose and pixels travel "
                     "together, so they can never be mismatched.",
            ha="center", fontsize=7.4, color=MUT, style="italic")
    _save(fig, "fig_packet")


if __name__ == "__main__":
    fig_frames()
    fig_raypl()
    fig_parallax()
    fig_flow()
    fig_packet()
    print("done")
