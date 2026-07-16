# if using a .ipynb in a folder, this heading is needed before importing
import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, project_root)

from src.poisson_hypergraph import GH
from matplotlib import pyplot as plt
import pandas as pd
import os 
import seaborn as sns
import xgi
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import statistics
import scripts.figure_settings as fs

sns.set_style("whitegrid")
fs.set_fonts()


simulated_annealing_ARI_averages = {}
for dataset in os.listdir("throughput/simulated_annealing"):
    if dataset != "senate_bills_warmstart":
        ari_average_data = []
        for filename in os.listdir(f"throughput/simulated_annealing/{dataset}/metrics"):
            
            if filename.endswith(".csv"):
                # df = pd.concat([df, pd.read_csv(f"throughput/community-structure-sims/{filename}")], ignore_index=True)
                ari_average_data.append(pd.read_csv(f"throughput/simulated_annealing/{dataset}/metrics/{filename}").iloc[-1]["ari"])

        simulated_annealing_ARI_averages[dataset] = statistics.mean(ari_average_data)
            
print(simulated_annealing_ARI_averages)

modularity_ari = {}
for dataset in os.listdir("throughput/modularity"):
    for filename in os.listdir(f"throughput/modularity/{dataset}/metrics"):
        if filename.endswith(".csv"):
            # df = pd.concat([df, pd.read_csv(f"throughput/community-structure-sims/{filename}")], ignore_index=True)
            modularity_ari[dataset] = pd.read_csv(f"throughput/modularity/{dataset}/metrics/{filename}").iloc[-1]["ari"]

print(modularity_ari)

spectral_ari = {}
for dataset in os.listdir("throughput/spectral-clustering"):
    for filename in os.listdir(f"throughput/spectral-clustering/{dataset}/metrics"):
        if filename.endswith(".csv"):
            # df = pd.concat([df, pd.read_csv(f"throughput/community-structure-sims/{filename}")], ignore_index=True)
            spectral_ari[dataset] = pd.read_csv(f"throughput/spectral-clustering/{dataset}/metrics/{filename}").iloc[-1]["ari"]

print(spectral_ari)



df = pd.DataFrame(
    {
        "Simulated Annealing": [simulated_annealing_ARI_averages['senate_bills'], simulated_annealing_ARI_averages['primaryschool'], simulated_annealing_ARI_averages['highschool']],
        "Modularity": [modularity_ari['senate_bills'], modularity_ari['primaryschool'], modularity_ari['highschool']],
        "Spectral Clustering": [spectral_ari['senate_bills'], spectral_ari['primaryschool'], spectral_ari['highschool']]
    },
    index = ["Senate Bills", "Primary School", "High School"]
)

print(df.T.to_latex(
    float_format="%.3f",
    bold_rows=True,
    caption="Community detection results.",
    label="tab:results"
))


import seaborn as sns
import matplotlib.pyplot as plt

# 1. Re-collect the individual ARI scores for the long-form DataFrame
sa_data = []

for dataset in ["senate_bills", "primaryschool", "highschool"]:
    folder_path = f"throughput/simulated_annealing/{dataset}/metrics"
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                df = pd.read_csv(f"{folder_path}/{filename}")
                ari_val = df.iloc[-1]["ari"]
                likelihood = df.iloc[-1]["likelihood"]
                sa_data.append({"Dataset": dataset, "ARI": ari_val, "Likelihood": likelihood})

sa_df = pd.DataFrame(sa_data)

best_likelihoods = sa_df.sort_values(["Dataset", "Likelihood"], ascending=False).drop_duplicates("Dataset")


name_mapper = {
    "senate_bills": "senate-bills",
    "primaryschool": "primary-school",
    "highschool": "high-school"
}

sa_df["Dataset"] = sa_df["Dataset"].map(name_mapper)



# 2. Define the order so the boxplot matches your hlines loop
order = ["senate-bills", "primary-school", "high-school"]



fig, ax = plt.subplots(figsize=(8.0, 5.0))

# Boxplot
sns.boxplot(
    data=sa_df,
    y="Dataset",
    x="ARI",
    order=order,
    color="0.85",
    width=0.55,
    linewidth=1,
    fliersize=7,
    ax=ax, 
    boxprops=dict(alpha=0.5), 
    flierprops=dict(alpha = 0.5),
    whiskerprops=dict(alpha = 0.5),     # 50% opaque whiskers
    capprops=dict(alpha = 0.5),         # 50% opaque caps
    medianprops=dict(alpha = 0.5)  
    
)

prior_methods = {"senate_bills" : {
    "nonbacktracking" : 0.25, 
    "hmod" : 0.09
},
                 "primaryschool" : {
    "nonbacktracking" : 0.15,
    "hmod" : np.nan
},
                 "highschool" : {
    "nonbacktracking" : 0.0,
    "hmod": np.nan
}
}




line_length = 0.25
# Overlay algorithm results
for i, dataset in enumerate(["senate_bills", "primaryschool", "highschool"]):
    
    
    ax.scatter(
        best_likelihoods[best_likelihoods["Dataset"] == dataset]["ARI"].values[0],
        i,
        color=fs.palette[0],
        s=100,
        zorder=500,
        label="CHILI SA\n(highest likelihood)" if i == 0 else None,
        edgecolors="black",
        linewidth=1.5,
        marker = "o"
    )
    
    
    if dataset == "senate_bills":
        ax.scatter(
            prior_methods[dataset]["hmod"],
            i,
            color=fs.palette[1],
            s=100,
            zorder=5,
            label="Hypergraph\nmodularity" if i == 0 else None,
            edgecolors="black",
            linewidth=1.5,
            marker = "P"
        )
    
    ax.scatter(
        prior_methods[dataset]["nonbacktracking"],
        i,
        color=fs.palette[2],
        s=100,
        zorder=5,
        label="Nonbacktracking\nspectral clustering" if i == 0 else None,
        edgecolors="black",
        linewidth=1.5,
        marker = "D"
    )
    
    # modularity
    ax.scatter(
        modularity_ari[dataset],
        i,
        facecolor="#B8B8B8",
        s=100,
        zorder=50000,
        label="Greedy\nmodularity" if i == 0 else None,
        edgecolors="black",
        linewidth=1.5,
        marker = "^"
    )
    # spectral 
    ax.scatter(
        spectral_ari[dataset],
        i,
        facecolor="#B8B8B8",
        s=100,
        zorder=5,
        label ="Spectral\nclustering" if i == 0 else None,
        edgecolors="black",
        linewidth=1.5,
        marker = "s"
    )
    
    
    

# Labels
ax.set_ylabel("")
ax.set_xlabel("Adjusted Rand Index (ARI)", fontsize=14)

# Remove unnecessary spines
# sns.despine(ax=ax)

# Legend beneath plot

reorder = lambda l, nc: sum((l[i::nc] for i in range(nc)), [])
h, l = ax.get_legend_handles_labels()
h = reorder(h, 3)
l = reorder(l, 3)

ax.legend(h, l, 
    loc="upper center",
    bbox_to_anchor=(0.5, -0.4),
    ncol=3,
    frameon=False,
    fontsize=14
)

# increase the fontsize of the y axis tick labels

ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)


plt.tight_layout()
plt.savefig("fig/community_detection_boxplot.pdf", dpi=300, bbox_inches="tight")
plt.savefig("fig/community_detection_boxplot.png", dpi=600, bbox_inches="tight")