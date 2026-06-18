from src.poisson_hypergraph import GH
from src.algorithms.sem import sem_functions
import xgi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import json
import os

# ── Configuration ────────────────────────────────────────────────────────────

DATASETS = [
    {
        "label":   "senate-bills",
        "json":    "throughput/senate_bills.json",
        "kwargs":  {"nodetype": int},
        "color":   "#B81365",
        "marker":  "o",
    },
    {
        "label":   "coauthorship",
        "json":    "throughput/gender_coauth_sorted.json",
        "kwargs":  {"nodetype": int},
        "color":   "#A7ACD9",
        "marker":  "s",
    },
    {
        "label":   "high-school",
        "json":    "throughput/highschool_gender.json",
        "kwargs":  {"nodetype": int, "edgetype": int},
        "color":   "#1B998B",
        "marker":  "^",
    },
    {
        "label":   "primary-school",
        "json":    "throughput/primaryschool_gender.json",
        "kwargs":  {"nodetype": int, "edgetype": int},
        "color":   "#1B998B",
        "marker":  "D",
    },
    {
        "label":   "house-bills",
        "json":    "throughput/house_bills.json",
        "kwargs":  {"nodetype": int},
        "color":   "#B81365",
        "marker":  "P",
    },
    {
        "label":   "emails",
        "json":    "throughput/emails.json",
        "kwargs":  {"nodetype": int},
        "color":   "#E28413",
        "marker":  "X",
    },
]

SEM_PARAMS = dict(
    s_initial       = np.array([1, 2, 1, 2, 0.5, 0.5, 0.5, 0.5]),
    iteration_limit = 8000,
    initial_rate    = 0.01,
    constant        = 0.001,
)

CACHE_FILE = "throughput/sem_final_estimates.json"

# ── Load cached estimates or run SEM ─────────────────────────────────────────

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return None

def save_cache(finals):
    # Strip non-serialisable keys (color, marker) before saving
    cache = []
    for f in finals:
        cache.append({k: v for k, v in f.items() if k not in ("color", "marker")})
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"Saved estimates → {CACHE_FILE}")

cached = load_cache()

if cached is not None:
    print(f"Loading cached SEM estimates from {CACHE_FILE}")
    # Re-attach plotting style from DATASETS config
    style = {ds["label"]: {"color": ds["color"], "marker": ds["marker"]} for ds in DATASETS}
    finals = []
    for entry in cached:
        finals.append({**entry, **style[entry["label"]]})
else:
    sem    = sem_functions()
    finals = []

    for ds in DATASETS:
        print(f"\n=== {ds['label']} ===")
        H  = xgi.read_json(ds["json"], **ds["kwargs"])
        gh = GH(H, [0, 1], 0, 0)

        estimates = sem.SEM_without_likelihood(
            gh,
            SEM_PARAMS["s_initial"].copy(),
            SEM_PARAMS["iteration_limit"],
            SEM_PARAMS["initial_rate"],
            SEM_PARAMS["constant"],
        )
        est = np.array(estimates)
        # columns: [iter, p, q, gam_nu, gam_nr, gam_eu, gam_er]
        _, p, q, gam_nu, gam_nr, gam_eu, gam_er = est[-1]

        finals.append({
            "label":   ds["label"],
            "color":   ds["color"],
            "marker":  ds["marker"],
            "p":       p,       # p̂₊
            "q":       q,       # p̂₋
            "gam_nu":  gam_nu,  # â₊
            "gam_nr":  gam_nr,  # â₋
            "gam_eu":  gam_eu,  # r̂₊
            "gam_er":  gam_er,  # r̂₋
        })
        print(f"  p̂₊={p:.4f}  p̂₋={q:.4f}  r̂₊={gam_eu:.4f}  r̂₋={gam_er:.4f}"
              f"  â₊={gam_nu:.4f}  â₋={gam_nr:.4f}")

    save_cache(finals)

# After finals is assembled (either from cache or SEM run), reorder to desired legend order
LEGEND_ORDER = ["senate-bills", "house-bills", "high-school", "primary-school", "coauthorship", "emails"]
finals = sorted(finals, key=lambda f: LEGEND_ORDER.index(f["label"]))
# ── Plot ─────────────────────────────────────────────────────────────────────

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   13,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  11,
})

PANELS = [
    {
        "title":  "Edge copy",
        "xlabel": r"$\hat{p}_+$",
        "ylabel": r"$\hat{p}_-$",
        "xkey":   "p",
        "ykey":   "q",
    },
    {
        "title":  "External node addition",
        "xlabel": r"$\hat{r}_+$",
        "ylabel": r"$\hat{r}_{-}$",
        "xkey":   "gam_eu",
        "ykey":   "gam_er",
    },
    {
        "title":  "Novel node addition",
        "xlabel": r"$\hat{a}_+$",
        "ylabel": r"$\hat{a}_{-}$",
        "xkey":   "gam_nu",
        "ykey":   "gam_nr",
    },
]

fig, axs = plt.subplots(1, 3, figsize=(11, 4.5), layout="constrained")
fig.set_constrained_layout_pads(wspace=0.08)

for ax, panel in zip(axs, PANELS):
    ax.set_title(panel["title"], pad=10)

    for f in finals:
        ax.scatter(
            f[panel["xkey"]], f[panel["ykey"]],
            color=f["color"],
            marker=f["marker"],
            s=120,
            alpha=0.7,
            zorder=5,
            label=f["label"],
            edgecolors="white",
            linewidths=0.8,
        )

    # axis limits with margin
    all_x = [f[panel["xkey"]] for f in finals]
    all_y = [f[panel["ykey"]] for f in finals]
    lo = min(min(all_x), min(all_y))
    hi = max(max(all_x), max(all_y))
    margin = (hi - lo) * 0.12 if hi != lo else 0.1
    lim = (lo - margin, hi + margin)
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    ax.plot(
        lim, lim,
        linestyle="--",
        color="0.5",
        linewidth=1.5,
        zorder=1,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

axs[0].set_xlabel(r"$\hat{\rho}_-$")
axs[0].set_ylabel(r"$\hat{\rho}_+$")

axs[1].set_xlabel(r"$\hat{\gamma}_-$")
axs[1].set_ylabel(r"$\hat{\gamma}_+$")

axs[2].set_xlabel(r"$\hat{\eta}_{-}$")
axs[2].set_ylabel(r"$\hat{\eta}_+$")

handles, labels = axs[0].get_legend_handles_labels()
axs[2].legend(
    handles, labels,
    loc="upper left",
    frameon=True,
    framealpha=0.9,
)

# ── after the scatter loop for axs[0], before fig.savefig ──

# Add homophily / heterophily arrows to the first panel only
# Homophily arrow
axs[0].annotate(
    "",
    xy=(0.42, 0.76),
    xytext=(0.58, 0.62),
    arrowprops=dict(
        arrowstyle="->",
        connectionstyle="arc3,rad=-0.35",
        color="black",
        lw=1.2,
    ),
)
axs[0].text(0.48, 0.7, "homophily", fontsize=10, ha="left", va="center")

# Heterophily arrow
axs[0].annotate(
    "",
    xy=(0.78, 0.45),
    xytext=(0.66, 0.5),
    arrowprops=dict(
        arrowstyle="->",
        connectionstyle="arc3,rad=-0.35",
        color="black",
        lw=1.2,
    ),
)
axs[0].text(0.52, 0.48, "heterophily", fontsize=10, ha="left", va="top")

fig.savefig("fig/empiricial_sem.png", dpi=300, bbox_inches="tight")