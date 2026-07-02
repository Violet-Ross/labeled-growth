import numpy as np
import csv
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns

import scripts.figure_settings as fs



sns.set_style("whitegrid")

fs.set_fonts()

# ── 1. True parameter values ──────────────────────────────────────────────────

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

# ── 2. KL divergence functions ────────────────────────────────────────────────

def kl_bernoulli(p, q):
    """KL(Bernoulli(p) ∥ Bernoulli(q))"""
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def kl_poisson(lam_p, lam_q):
    """KL(Poisson(lam_p) ∥ Poisson(lam_q))"""
    eps = 1e-12
    lam_p = np.maximum(lam_p, eps)
    lam_q = np.maximum(lam_q, eps)
    return lam_p * np.log(lam_p / lam_q) - lam_p + lam_q

def total_kl(truth, estimate):
    """
    Sum of KL(truth ∥ estimate) across all 6 parameters.
    Parameters 0-1: Bernoulli means
    Parameters 2-5: Poisson means
    """
    kl = 0.0
    for i in range(2):
        kl += kl_bernoulli(truth[i], estimate[i])
    for i in range(2, 6):
        kl += kl_poisson(truth[i], estimate[i])
    return kl

# ── 3. Labels ─────────────────────────────────────────────────────────────────

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

# ── Bracket groupings ─────────────────────────────────────────────────────────

y_bracket_groups = [
    ("Strength of homophily", 0, 5),
    ("Strength of mechanisms", 6, 9),
]

x_bracket_groups = [
    ("Strength of \n homophily", 0, 1),
    ("Strength of \n copy mechanism", 2, 3),
]

# ── 4. Load CSV results and compute average KL divergence across iterations ───

INPUT_DIR = "throughput/synthetic_results"
N_ITERS = 20

n_rows, n_cols = 10, 4
kl_values = np.full((n_rows, n_cols), np.nan)

# Remap: original row index → display row index
# Original row 9 (graphs 37-40) → display row 6
# Original rows 6, 7, 8 → display rows 7, 8, 9
original_to_display_row = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
    9: 6,
    6: 7, 7: 8, 8: 9,
}


kl_file_max = ""
kl_file_min = ""

for graph_idx, true_theta in enumerate(true_thetas, start=1):
    original_row = (graph_idx - 1) // n_cols
    col = (graph_idx - 1) % n_cols
    row = original_to_display_row[original_row]
    truth = np.array(true_theta)

    iter_kls = []
    for j in range(1, N_ITERS + 1):
        csv_path = os.path.join(INPUT_DIR, f"graph{graph_idx}_rep{j}_sem_ests_extended.csv")
        if not os.path.exists(csv_path):
            print(f"  [WARNING] Missing: {csv_path}")
            continue

        with open(csv_path, newline="") as f:
            all_rows = list(csv.reader(f))

        last_row = [float(v) for v in all_rows[-1] if v.strip() != ""][1:]
        estimate = np.array(last_row)
        kl = total_kl(truth, estimate)
        
        iter_kls.append(kl)

    if iter_kls:
        mean_kl = np.mean(iter_kls)
        
        if mean_kl > np.nanmax(kl_values):
            kl_file_max = csv_path
            kl_true_theta_max = truth
            max_kl = mean_kl
        if mean_kl < np.nanmin(kl_values):
            kl_file_min = csv_path
            kl_true_theta_min = truth
            min_kl = mean_kl

        kl_values[row, col] = mean_kl
        
        
        
        

kl_values = kl_values[:, [0, 1, 3, 2]]


print(f"File with maximum KL divergence: {kl_file_max}")
print(f"Maximum KL divergence: {max_kl}")
print(f"File with minimum KL divergence: {kl_file_min}")
print(f"Minimum KL divergence: {min_kl}")

# ── 5. Plot ───────────────────────────────────────────────────────────────────



fig = plt.figure(layout = "constrained", figsize=(12, 6))
gs = GridSpec(2, 3, figure=fig)

ax = fig.add_subplot(gs[:2, 0])

# fig, ax = plt.subplots(figsize=(8, 8))
# plt.subplots_adjust(left=0.45, top=0.75)

pink_gradient = LinearSegmentedColormap.from_list("pink_gradient", ["#fce8f1", "#e06098", "#B81365"])
im = ax.imshow(kl_values, cmap=pink_gradient, aspect="auto")

ax.grid(False)

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("KL divergence", fontsize=10)

ax.set_xticks(range(n_cols))
ax.set_xticklabels(col_labels, fontsize=10, rotation=0, ha="center")
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()
ax.set_yticks(range(n_rows))
ax.set_yticklabels(row_labels, fontsize=10)
ax.yaxis.tick_left()

vmin_idx = np.unravel_index(np.nanargmin(kl_values), kl_values.shape)
vmax_idx = np.unravel_index(np.nanargmax(kl_values), kl_values.shape)

# Visual guides for dividers
linewidth = 4
ax.vlines(x=1.5, ymin=-0.5, ymax=n_rows-0.5, colors="white", linestyles="solid", linewidth=linewidth)
ax.hlines(y=1.5, xmin=-0.5, xmax=n_cols-0.5, colors="white", linestyles="solid", linewidth=linewidth)
ax.hlines(y=3.5, xmin=-0.5, xmax=n_cols-0.5, colors="white", linestyles="solid", linewidth=linewidth)
ax.hlines(y=5.5, xmin=-0.5, xmax=n_cols-0.5, colors="white", linestyles="solid", linewidth=linewidth)



for i, idx in enumerate([vmin_idx, vmax_idx]):
    r, c = idx
    val = kl_values[r, c]
    ax.text(c, r, f"{val:.3f}", ha="center", va="center",
            fontsize=10, color=["black", "white"][i])

# ── 6. Y-axis brackets (left side) ───────────────────────────────────────────

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

# ── 6b. Y-axis subgroup brackets ─────────────────────────────────────────────

sub_bracket_x = -0.33
sub_cap_width  =  0.04
sub_text_x     = -0.39

y_sub_bracket_groups = [
    ("External node \n addition", 0, 1),
    ("Novel node \n addition", 2, 3),
    ("Both", 4, 5),
]

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

ax.annotate("(a)", xy=(-0.2, 1.05), xycoords='axes fraction', fontsize=16, fontweight='bold', bbox=dict(facecolor='white', edgecolor='white', alpha=0.5))

# ── 7. X-axis brackets (top side) ────────────────────────────────────────────

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



print(np.nanmin(kl_values), np.nanmax(kl_values))





pixel_1_ix = [1,2]
pixel_2_ix = [3,4]

labs = ["(b)", "(c)", "(d)", "(e)"]


for k in range(2): 
    
    df = pd.read_csv([kl_file_min, kl_file_max][k])
    
    true_theta = [kl_true_theta_min, kl_true_theta_max][k]
    pixel_ix = [pixel_1_ix, pixel_2_ix][k]
    
    n_steps = len(df)
    
    
    
    lil_ax = fig.add_subplot(gs[k, 1])
    lil_ax.set_ylabel(f"Parameter value")
    lil_ax.plot(df.iloc[:,1], color=fs.palette[0], label=r"$\rho_{+}$", linewidth = 3)
    lil_ax.plot(df.iloc[:,2], color=fs.lighten(fs.palette[0]), label=r"$\rho_{-}$", linewidth = 3)
    lines_1, labs_1 = lil_ax.get_legend_handles_labels()
    
    lil_ax.plot([0, n_steps], [true_theta[0], true_theta[0]], color=fs.palette[0], linestyle='--')
    lil_ax.plot([0, n_steps], [true_theta[1], true_theta[1]], color=fs.lighten(fs.palette[0]), linestyle='--')
    
    lil_ax.plot()
    
    lil_ax.set_title(f"Node copy parameters" if k == 0 else "")
    lil_ax.set_xlabel(f"SEM timestep" if k == 1 else "")
    if k == 0:
        lil_ax.set_xticklabels([])
        
    lil_ax.semilogx()
    lil_ax.set_ylim(0, 1)
    
    if k == 1:
        # two column legend
        lil_ax.legend(lines_1, labs_1, loc='upper right', fontsize=11, ncol=2)
        
    if k == 0: 
        lil_ax.set_xticklabels([])
        
    lil_ax.annotate(labs[2*k], xy=(0.05, 0.7), xycoords='axes fraction', fontsize=16, fontweight='bold')
    
    
    
    lil_ax = fig.add_subplot(gs[k, 2])
    lil_ax.plot(df.iloc[:,3], color=fs.palette[1], label = r"$\gamma_{+}$", linewidth = 3)
    lil_ax.plot(df.iloc[:,5], color=fs.palette[2], label = r"$\eta_{+}$", linewidth = 3)
    lil_ax.plot(df.iloc[:,4], color=fs.lighten(fs.palette[1]), label = r"$\gamma_{-}$", linewidth = 3)
    lil_ax.plot(df.iloc[:,6], color=fs.lighten(fs.palette[2]), label = r"$\eta_{-}$", linewidth = 3)

    lil_ax.plot([0, n_steps], [true_theta[2], true_theta[2]], color=fs.palette[1], linestyle='--')
    lil_ax.plot([0, n_steps], [true_theta[3], true_theta[3]], color=fs.lighten(fs.palette[1]), linestyle='--')
    lil_ax.plot([0, n_steps], [true_theta[4], true_theta[4]], color=fs.palette[2], linestyle='--')
    lil_ax.plot([0, n_steps], [true_theta[5], true_theta[5]], color=fs.lighten(fs.palette[2]), linestyle='--')
    lines_2, labs_2 = lil_ax.get_legend_handles_labels()
    
    lil_ax.set_ylim(0, 2.1)
    
    print(true_theta)

    lil_ax.set_xlabel(f"SEM timestep" if k == 1 else "")
    lil_ax.set_title(f"Node addition parameters" if k == 0 else "")
    if k == 0:
        lil_ax.set_xticklabels([])
        
    lil_ax.semilogx()
    
    if k == 1:
        # two column legend
        lil_ax.legend(lines_2, labs_2, loc='upper right', fontsize=11, ncol=2)
        
    if k == 0: 
        lil_ax.set_xticklabels([])

    lil_ax.annotate(labs[2*k+1], xy=(0.05, 0.8), xycoords='axes fraction', fontsize=16, fontweight='bold', bbox=dict(facecolor='white', edgecolor='white', alpha=0.5))

output_path = "fig/sem_convergence_heatmap.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved → {output_path}")

