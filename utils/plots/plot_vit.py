import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter, LogLocator, NullFormatter
from pathlib import Path
import numpy as np

# ============================================================
# Configuration
# ============================================================
CSV_PATH = "/Users/luiszuin/Desktop/Polytechnique/3AP2/multimodal_gen_ai/chess_classifier_vit_v2_stats_history.csv"
OUT_PNG = "vit_error_rate_evolution.png"
OUT_PDF = "vit_error_rate_evolution.pdf"

# Poster-inspired colors (same palette as previous plot)
COLOR_BLUE = "#0B4F7D"
COLOR_ORANGE = "#E28B2D"
COLOR_GRID = "#D9E1E8"
COLOR_TEXT = "#1F2D3D"
COLOR_SPINE = "#6E7B87"
COLOR_BG = "#FFFFFF"

# ============================================================
# Read data
# CSV columns observed:
# stage, epoch, epoch_progress, tag, train_loss, train_acc,
# val_loss_noaug, val_acc_noaug, val_loss_aug, val_acc_aug
# ============================================================
df = pd.read_csv(CSV_PATH)

# Clean column names
df.columns = [str(c).strip() for c in df.columns]

required_cols = ["epoch_progress", "val_acc_noaug", "val_acc_aug"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Keep only rows with validation values
df = df.dropna(subset=required_cols).copy()

# Sort by training progression
df = df.sort_values("epoch_progress")

# ============================================================
# X axis: continuous epoch progression
# Example:
# 0.333, 0.667, 1.000, 1.333, ...
# We shift by +1 if you want the plot to visually start at epoch 1
# ============================================================
x = df["epoch_progress"].to_numpy()

# If your logs start from epoch 0, but you want the plot labeled from 1:
x = x + 1

# ============================================================
# Convert validation accuracy to error rate (%)
# error_rate = (1 - accuracy) * 100
# ============================================================
err_clean = (1.0 - df["val_acc_noaug"].to_numpy()) * 100.0
err_aug   = (1.0 - df["val_acc_aug"].to_numpy()) * 100.0

# Avoid log(0) if a value is ever exactly zero
EPS = 1e-4
err_clean = np.clip(err_clean, EPS, None)
err_aug   = np.clip(err_aug, EPS, None)

# ============================================================
# Figure style
# ============================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
})

fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=300)
fig.patch.set_facecolor(COLOR_BG)
ax.set_facecolor(COLOR_BG)

# ============================================================
# Curves
# ============================================================
line_clean, = ax.plot(
    x, err_clean,
    color=COLOR_BLUE,
    linewidth=2.8,
    marker="o",
    markersize=6.2,
    markerfacecolor=COLOR_BLUE,
    markeredgecolor="white",
    markeredgewidth=1.0,
    label="Clean validation",
    zorder=3,
)

line_aug, = ax.plot(
    x, err_aug,
    color=COLOR_ORANGE,
    linewidth=2.8,
    marker="s",
    markersize=6.0,
    markerfacecolor=COLOR_ORANGE,
    markeredgecolor="white",
    markeredgewidth=1.0,
    label="Augmented validation",
    zorder=3,
)

# ============================================================
# Axes
# ============================================================
ax.set_xlabel("Epoch", weight= "bold")
ax.set_ylabel("Validation Error Rate (%)", color=COLOR_TEXT, weight="bold")

# Log scale on y
ax.set_yscale("log")

# Good x limits
ax.set_xlim(x.min() - 0.1, x.max() + 0.2)

# Smart y limits with a bit of padding
ymin = min(err_clean.min(), err_aug.min())
ymax = max(err_clean.max(), err_aug.max())
ax.set_ylim(ymin * 0.75, ymax * 1.25)

# Integer epoch ticks
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# Nice percent formatting
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.2g}%"))

# Minor ticks for log scale
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
ax.yaxis.set_minor_formatter(NullFormatter())

# Tick colors
ax.tick_params(axis="x", colors=COLOR_TEXT)
ax.tick_params(axis="y", colors=COLOR_TEXT)

# Grid: major only to keep it elegant
ax.grid(True, axis="both", which="major", linestyle="--", linewidth=0.8, color=COLOR_GRID, alpha=0.9)
ax.grid(True, axis="y", which="minor", linestyle=":", linewidth=0.5, color=COLOR_GRID, alpha=0.55)
ax.set_axisbelow(True)

# Spines
for spine in ["top", "bottom", "left", "right"]:
    ax.spines[spine].set_color(COLOR_SPINE)
    ax.spines[spine].set_linewidth(1.0)

# ============================================================
# Legend
# ============================================================
legend = ax.legend(
    loc="upper right",
    frameon=True,
    fancybox=True,
    framealpha=0.96,
    borderpad=0.6,
)
legend.get_frame().set_edgecolor("#C9D3DC")
legend.get_frame().set_linewidth(0.9)

# ============================================================
# Final value annotations
# ============================================================
ax.annotate(
    f"{err_clean[-1]:.3f}%",
    xy=(x[-1], err_clean[-1]),
    xytext=(10, 6),
    textcoords="offset points",
    ha="right",
    color=COLOR_BLUE,
    fontsize=10,
    weight="bold"
)

ax.annotate(
    f"{err_aug[-1]:.2f}%",
    xy=(x[-1], err_aug[-1]),
    xytext=(10, -14),
    textcoords="offset points",
    ha="right",
    color=COLOR_ORANGE,
    fontsize=10,
    weight="bold"
)

# ============================================================
# Layout and save
# ============================================================
plt.tight_layout()

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor=COLOR_BG)
fig.savefig(OUT_PDF, bbox_inches="tight", facecolor=COLOR_BG)

print(f"Saved: {Path(OUT_PNG).resolve()}")
print(f"Saved: {Path(OUT_PDF).resolve()}")

plt.show()