import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
EXCEL_PATH = "/Users/luiszuin/Desktop/Polytechnique/3AP2/multimodal_gen_ai/project_multimodal_gen_ai/encoder_evolution.xlsx"   # <- your input file
SHEET_NAME = 0                          # can also be a sheet name string
OUT_PNG = "encoder_accuracy_elo_evolution.png"
OUT_PDF = "encoder_accuracy_elo_evolution.pdf"

# Poster-inspired colors
COLOR_BLUE = "#0B4F7D"      # main poster/header blue
COLOR_ORANGE = "#E28B2D"    # accent orange
COLOR_GRID = "#D9E1E8"
COLOR_TEXT = "#1F2D3D"
COLOR_SPINE = "#6E7B87"
COLOR_BG = "#FFFFFF"

# ============================================================
# Read data
# Expected columns:
#   Epoch | Val Accuracy | Estimated ELO
# ============================================================
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

# Clean column names in case of extra spaces
df.columns = [str(c).strip() for c in df.columns]

required_cols = ["Epoch", "Val Accuracy", "Estimated ELO"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[required_cols].copy()
df = df.sort_values("Epoch")

x = df["Epoch"]
val_acc = df["Val Accuracy"]
elo = df["Estimated ELO"]

# ============================================================
# Helper for visually good scaling
# ============================================================
def padded_limits(series, pad_ratio=0.08, min_pad=1e-3):
    smin = float(series.min())
    smax = float(series.max())
    span = max(smax - smin, min_pad)
    pad = span * pad_ratio
    return smin - pad, smax + pad

# Accuracy is usually very tight, so slightly larger relative padding helps visually
acc_ymin, acc_ymax = padded_limits(val_acc, pad_ratio=0.15, min_pad=0.005)
elo_ymin, elo_ymax = padded_limits(elo, pad_ratio=0.10, min_pad=25)

# ============================================================
# Figure
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

fig, ax1 = plt.subplots(figsize=(7.4, 4.6), dpi=300)
fig.patch.set_facecolor(COLOR_BG)
ax1.set_facecolor(COLOR_BG)

# Left axis: ELO
line_elo, = ax1.plot(
    x, elo,
    color=COLOR_BLUE,
    linewidth=2.8,
    marker="o",
    markersize=6.5,
    markerfacecolor=COLOR_BLUE,
    markeredgecolor="white",
    markeredgewidth=1.0,
    label="Estimated ELO",
    zorder=3,
)

# Right axis: validation accuracy
ax2 = ax1.twinx()
line_acc, = ax2.plot(
    x, val_acc,
    color=COLOR_ORANGE,
    linewidth=2.8,
    marker="s",
    markersize=6.0,
    markerfacecolor=COLOR_ORANGE,
    markeredgecolor="white",
    markeredgewidth=1.0,
    label="Validation Accuracy",
    zorder=3,
)

# ============================================================
# Axes labels and limits
# ============================================================
ax1.set_xlabel("Epoch")
ax1.set_ylabel("UCI ELO", color=COLOR_BLUE, weight="bold")
ax2.set_ylabel("Validation Accuracy", color=COLOR_ORANGE, weight="bold")

ax1.set_xlim(x.min() - 0.2, x.max() + 0.2)
ax1.set_ylim(elo_ymin, elo_ymax)
ax2.set_ylim(acc_ymin, acc_ymax)

# Force integer ticks on epochs
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

# Nice tick formatting
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(round(v))}"))
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.3f}"))

# Tick colors
ax1.tick_params(axis="y", colors=COLOR_BLUE)
ax2.tick_params(axis="y", colors=COLOR_ORANGE)
ax1.tick_params(axis="x", colors=COLOR_TEXT)

# Grid
ax1.grid(True, axis="both", linestyle="--", linewidth=0.8, color=COLOR_GRID, alpha=0.9)
ax1.set_axisbelow(True)

# Spines
for spine in ["top", "bottom", "left", "right"]:
    ax1.spines[spine].set_color(COLOR_SPINE)
    ax1.spines[spine].set_linewidth(1.0)

ax2.spines["top"].set_color(COLOR_SPINE)
ax2.spines["right"].set_color(COLOR_SPINE)
ax2.spines["right"].set_linewidth(1.0)

# ============================================================
# Title
# ============================================================
# ax1.set_title(
#     "Encoder checkpoint evolution: validation accuracy vs tournament ELO",
#     color=COLOR_TEXT,
#     weight="bold",
#     pad=12
# )

# ============================================================
# Legend
# ============================================================
handles = [line_elo, line_acc]
labels = [h.get_label() for h in handles]
legend = ax1.legend(
    handles, labels,
    loc="upper left",
    frameon=True,
    fancybox=True,
    framealpha=0.96,
    borderpad=0.6,
)
legend.get_frame().set_edgecolor("#C9D3DC")
legend.get_frame().set_linewidth(0.9)

# ============================================================
# Optional subtle value annotations on last points
# ============================================================
# ax1.annotate(
#     f"{elo.iloc[-1]:.0f}",
#     xy=(x.iloc[-1], elo.iloc[-1]),
#     xytext=(8, 8),
#     textcoords="offset points",
#     color=COLOR_BLUE,
#     fontsize=10,
#     weight="bold"
# )

# ax2.annotate(
#     f"{val_acc.iloc[-1]:.3f}",
#     xy=(x.iloc[-1], val_acc.iloc[-1]),
#     xytext=(8, -16),
#     textcoords="offset points",
#     color=COLOR_ORANGE,
#     fontsize=10,
#     weight="bold"
# )

ax1.annotate(
    f"{elo.iloc[-1]:.0f}",
    xy=(x.iloc[-1], elo.iloc[-1]),
    xytext=(-23, 8),
    textcoords="offset points",
    color=COLOR_BLUE,
    fontsize=10,
    weight="bold"
)

ax2.annotate(
    f"{val_acc.iloc[-1]:.3f}",
    xy=(x.iloc[-1], val_acc.iloc[-1]),
    xytext=(-25, -16),
    textcoords="offset points",
    color=COLOR_ORANGE,
    fontsize=10,
    weight="bold"
)

# ============================================================
# Layout and save
# ============================================================
plt.tight_layout()

# Save both with transparent=False for poster consistency
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor=COLOR_BG)
fig.savefig(OUT_PDF, bbox_inches="tight", facecolor=COLOR_BG)

print(f"Saved: {Path(OUT_PNG).resolve()}")
print(f"Saved: {Path(OUT_PDF).resolve()}")

plt.show()