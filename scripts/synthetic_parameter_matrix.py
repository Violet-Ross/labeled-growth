import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    import scripts.figure_settings as fs
    fs.set_fonts()
except Exception:
    fs = None

# ── 1. True parameter values (identical to complete_SEM_convergence_plots.py) ─

weak_cop       = [0.6, 0.4]
strong_cop     = [0.9, 0.2]
none_high_cop  = [0.9, 0.9]
none_low_cop   = [0.1, 0.1]

null_ext       = [0.5, 0.5]
weak_ext       = [1,   0.8]
strong_ext     = [2,   0.2]
none_high_ext  = [2,   2  ]
none_low_ext   = [0.1, 0.1]

null_nov       = [0.5, 0.5]
weak_nov       = [1,   0.8]
strong_nov     = [2,   0.2]
none_high_nov  = [2,   2  ]
none_low_nov   = [0.1, 0.1]

true_thetas = [
    weak_cop + weak_ext + null_nov,
    strong_cop + weak_ext + null_nov,
    none_high_cop + weak_ext + null_nov,
    none_low_cop + weak_ext + null_nov,
    weak_cop + strong_ext + null_nov,
    strong_cop + strong_ext + null_nov,
    none_high_cop + strong_ext + null_nov,
    none_low_cop + strong_ext + null_nov,
    weak_cop + null_ext + weak_nov,
    strong_cop + null_ext + weak_nov,
    none_high_cop + null_ext + weak_nov,
    none_low_cop + null_ext + weak_nov,
    weak_cop + null_ext + strong_nov,
    strong_cop + null_ext + strong_nov,
    none_high_cop + null_ext + strong_nov,
    none_low_cop + null_ext + strong_nov,
    weak_cop + weak_ext + weak_nov,
    strong_cop + weak_ext + weak_nov,
    none_high_cop + weak_ext + weak_nov,
    none_low_cop + weak_ext + weak_nov,
    weak_cop + strong_ext + strong_nov,
    strong_cop + strong_ext + strong_nov,
    none_high_cop + strong_ext + strong_nov,
    none_low_cop + strong_ext + strong_nov,
    weak_cop + none_high_ext + null_nov,
    strong_cop + none_high_ext + null_nov,
    none_high_cop + none_high_ext + null_nov,
    none_low_cop + none_high_ext + null_nov,
    weak_cop + null_ext + none_high_nov,
    strong_cop + null_ext + none_high_nov,
    none_high_cop + null_ext + none_high_nov,
    none_low_cop + null_ext + none_high_nov,
    weak_cop + none_high_ext + none_high_nov,
    strong_cop + none_high_ext + none_high_nov,
    none_high_cop + none_high_ext + none_high_nov,
    none_low_cop + none_high_ext + none_high_nov,
    weak_cop + none_low_ext + none_low_nov,
    strong_cop + none_low_ext + none_low_nov,
    none_high_cop + none_low_ext + none_low_nov,
    none_low_cop + none_low_ext + none_low_nov,
]

# ── 2. Labels (identical to original) ──────────────────────────────────────────

row_labels = [
    "Weak",
    "Strong",
    "Weak",
    "Strong",
    "Weak",
    "Strong",
    "Both mechanisms \n weak",
    "Strong external \n node mechanism",
    "Strong novel \n node mechanism",
    "Both mechanisms \n strong",
]

col_labels = [
    "Weak",
    "Strong",
    "Weak",
    "Strong",
]

y_bracket_groups = [
    ("Strength of homophily", 0, 5),
    ("Strength of mechanisms", 6, 9),
]

x_bracket_groups = [
    ("Strength of \n homophily", 0, 1),
    ("Strength of \n copy mechanism", 2, 3),
]

y_sub_bracket_groups = [
    ("External node \n addition", 0, 1),
    ("Novel node \n addition", 2, 3),
    ("Both", 4, 5),
]

# ── 3. Build a grid of parameter-value text, using the same row/col remap ─────
#     used in the original script so the layout lines up cell-for-cell.

n_rows, n_cols = 10, 4

original_to_display_row = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
    9: 6,
    6: 7, 7: 8, 8: 9,
}

def fmt(v):
    return f"{v:.2f}"

# text grid holds a formatted, multi-line string of (copy, extant, novel) params
text_grid = np.empty((n_rows, n_cols), dtype=object)

for graph_idx, true_theta in enumerate(true_thetas, start=1):
    original_row = (graph_idx - 1) // n_cols
    col = (graph_idx - 1) % n_cols
    row = original_to_display_row[original_row]

    rho_p, rho_m, gam_p, gam_m, eta_p, eta_m = true_theta

    cell_text = (
        rf"$\rho$: {fmt(rho_p)}, {fmt(rho_m)}" + "\n"
        rf"$\gamma$: {fmt(gam_p)}, {fmt(gam_m)}" + "\n"
        rf"$\eta$: {fmt(eta_p)}, {fmt(eta_m)}"
    )
    text_grid[row, col] = cell_text

# match the column reorder applied in the original script: [0, 1, 3, 2]
text_grid = text_grid[:, [0, 1, 3, 2]]

# ── 4. Plot ────────────────────────────────────────────────────────────────────
# Figure / GridSpec setup mirrors complete_SEM_convergence_plots.py exactly
# (same figsize and same 2x3 GridSpec, with the heatmap in gs[:2, 0]) so that
# this standalone heatmap has identical dimensions to panel (a) of the
# original figure.

fig = plt.figure(layout="constrained", figsize=(12, 6))
gs = GridSpec(2, 3, figure=fig)

ax = fig.add_subplot(gs[:2, 0])

# All-white "image" (same shape/aspect call as the original imshow) so the
# axes geometry, tick placement, and cell size match the original exactly.
white_img = np.ones((n_rows, n_cols, 3))
ax.imshow(white_img, aspect="auto")

ax.grid(False)

ax.set_xticks(range(n_cols))
ax.set_xticklabels(col_labels, fontsize=10, rotation=0, ha="center")
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()
ax.set_yticks(range(n_rows))
ax.set_yticklabels(row_labels, fontsize=10)
ax.yaxis.tick_left()

# Thin gray lines around every cell so cells are still legible on white
for x in np.arange(-0.5, n_cols, 1):
    ax.vlines(x=x, ymin=-0.5, ymax=n_rows - 0.5, colors="lightgray", linewidth=0.75)
for y in np.arange(-0.5, n_rows, 1):
    ax.hlines(y=y, xmin=-0.5, xmax=n_cols - 0.5, colors="lightgray", linewidth=0.75)

# Thicker section dividers, now in black/gray since white no longer shows on white
linewidth = 2.5
ax.vlines(x=1.5, ymin=-0.5, ymax=n_rows - 0.5, colors="black", linestyles="solid", linewidth=linewidth)
ax.hlines(y=1.5, xmin=-0.5, xmax=n_cols - 0.5, colors="black", linestyles="solid", linewidth=linewidth)
ax.hlines(y=3.5, xmin=-0.5, xmax=n_cols - 0.5, colors="black", linestyles="solid", linewidth=linewidth)
ax.hlines(y=5.5, xmin=-0.5, xmax=n_cols - 0.5, colors="black", linestyles="solid", linewidth=linewidth)

# Parameter-value text in every cell
for r in range(n_rows):
    for c in range(n_cols):
        txt = text_grid[r, c]
        if txt is not None:
            ax.text(c, r, txt, ha="center", va="center", fontsize=6.5, color="black")

# ── 5. Y-axis brackets (left side) — identical to original ────────────────────

bracket_x = -0.63
cap_width  =  0.04
text_x     = -0.69

trans = ax.transAxes

for label, r_start, r_end in y_bracket_groups:
    y_top    = 1 - (r_start / n_rows) - 0.5 / n_rows
    y_bottom = 1 - (r_end   / n_rows) - 0.5 / n_rows
    y_mid    = (y_top + y_bottom) / 2

    ax.plot([bracket_x, bracket_x], [y_bottom, y_top],
            color="black", lw=1.5, transform=trans, clip_on=False)
    ax.plot([bracket_x, bracket_x + cap_width], [y_top, y_top],
            color="black", lw=1.5, transform=trans, clip_on=False)
    ax.plot([bracket_x, bracket_x + cap_width], [y_bottom, y_bottom],
            color="black", lw=1.5, transform=trans, clip_on=False)
    ax.text(text_x, y_mid, label, ha="center", va="center",
            fontsize=9, transform=trans, rotation=90, clip_on=False)

sub_bracket_x = -0.33
sub_cap_width  =  0.04
sub_text_x     = -0.39

for label, r_start, r_end in y_sub_bracket_groups:
    y_top    = 1 - (r_start / n_rows) - 0.5 / n_rows
    y_bottom = 1 - (r_end   / n_rows) - 0.5 / n_rows
    y_mid    = (y_top + y_bottom) / 2

    ax.plot([sub_bracket_x, sub_bracket_x], [y_bottom, y_top],
            color="black", lw=1.5, transform=trans, clip_on=False)
    ax.plot([sub_bracket_x, sub_bracket_x + sub_cap_width], [y_top, y_top],
            color="black", lw=1.5, transform=trans, clip_on=False)
    ax.plot([sub_bracket_x, sub_bracket_x + sub_cap_width], [y_bottom, y_bottom],
            color="black", lw=1.5, transform=trans, clip_on=False)
    ax.text(sub_text_x, y_mid, label, ha="center", va="center",
            fontsize=9, transform=trans, rotation=90, clip_on=False)

# ── 6. X-axis brackets (top side) — identical to original ─────────────────────

bracket_y = 1.12
cap_height = 0.04
text_y     = 1.15

for label, c_start, c_end in x_bracket_groups:
    x_left  = (c_start / n_cols) + 0.5 / n_cols
    x_right = (c_end   / n_cols) + 0.5 / n_cols
    x_mid   = (x_left + x_right) / 2

    ax.plot([x_left, x_right], [bracket_y, bracket_y],
            color="black", lw=1.5, transform=trans, clip_on=False)
    ax.plot([x_left,  x_left],  [bracket_y, bracket_y - cap_height],
            color="black", lw=1.5, transform=trans, clip_on=False)
    ax.plot([x_right, x_right], [bracket_y, bracket_y - cap_height],
            color="black", lw=1.5, transform=trans, clip_on=False)
    ax.text(x_mid, text_y, label, ha="center", va="bottom",
            fontsize=9, transform=trans, clip_on=False)

output_path = "fig/sem_true_params_heatmap.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved → {output_path}")