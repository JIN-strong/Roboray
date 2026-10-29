"""QC plots: reprojection error curve (flagged frames recognizable; no fake plot when
there is no data).

Text on the plot is rendered in CJK when a CJK font is available; without one it
falls back to ASCII labels. Both paths must be free of missing-glyph warnings —
never leave a row of tofu boxes (□□□) masquerading as readable labels.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["plot_reprojection_curve"]

# Candidate CJK fonts (in priority order): first hit wins. Probed by the family
# names actually registered with font_manager — only registered fonts are really used
# by the Agg backend for rasterization.
_CJK_CANDIDATES = (
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Serif CJK SC",
    "Source Han Sans SC", "Microsoft YaHei", "SimHei",
    "Alibaba PuHuiTi", "Alibaba PuHuiTi 2.0", "Alibaba PuHuiTi 3.0",
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "PingFang SC",
    "Hiragino Sans GB", "SimSun", "simsun",
)


def _resolve_cjk_font() -> str | None:
    """Probe for an available CJK font family; None when none is found (the caller
    falls back to ASCII labels)."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for fam in _CJK_CANDIDATES:
        if fam in available:
            return fam
    return None


# Resolve once at module level: when a font is found, set rcParams globally so every
# plot from this module renders text without missing glyphs.
_PLOT_FONT = _resolve_cjk_font()
if _PLOT_FONT:
    plt.rcParams["font.sans-serif"] = [_PLOT_FONT, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _labels() -> dict:
    """Labels for the plot; always English, and CJK-rendered when a CJK font is
    available (both paths must be free of missing-glyph warnings)."""
    if _PLOT_FONT:
        return {"left": "Left", "right": "Right", "flag": "flagged (untrusted/filled)",
                "xlabel": "Frame", "ylabel": "Reproj error (px)"}
    return {"left": "Left", "right": "Right", "flag": "flagged (untrusted/filled)",
            "xlabel": "frame", "ylabel": "reproj err (px)"}


def plot_reprojection_curve(reprojection_error, out_png, *, flag_value: float = 1e6) -> dict:
    err = np.asarray(reprojection_error, np.float64)
    if err.ndim != 2 or err.size == 0:
        return {"plotted": False, "reason": "no reprojection error data"}
    trusted = np.isfinite(err) & (err < flag_value)
    if not trusted.any():
        return {"plotted": False, "reason": "all frames are flagged values (nothing to compare)"}
    t, hands = err.shape
    labels = _labels()
    fig, ax = plt.subplots(figsize=(6, 2.4), dpi=100)
    names = [labels["left"], labels["right"]]
    for h in range(hands):
        y = np.where(trusted[:, h], err[:, h], np.nan)
        ax.plot(np.arange(t), y, linewidth=1.0, label=names[h] if h < len(names) else f"hand{h}")
        flagged = np.where(~trusted[:, h])[0]
        if flagged.size:
            ax.scatter(flagged, np.zeros_like(flagged), marker="x", s=12,
                       color="red", label=labels["flag"] if h == 0 else None)
    ax.set_xlabel(labels["xlabel"]); ax.set_ylabel(labels["ylabel"])
    ax.legend(fontsize=7)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return {"plotted": True, "out_png": str(out_png),
            "trusted_frames": int(trusted.sum()), "flagged_frames": int((~trusted).sum())}
