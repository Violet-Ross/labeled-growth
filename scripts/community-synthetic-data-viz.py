import sys
import os
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
# read all csvs in throughput/community/synthetic and create a dataframe with all the data

df = pd.DataFrame()
for file in os.listdir('./throughput/community/synthetic'):
    if file.endswith('.csv'):
        df_temp = pd.read_csv('./throughput/community/synthetic/' + file)
        df = pd.concat([df, df_temp], ignore_index=True)
        

sns.set_style("whitegrid")
fig, axarr = plt.subplots(2, 3, figsize=(11, 6))

for i, vary in enumerate(["eta", "lambda", "gamma"]): 
    for j, metric in enumerate(["simulated_annealing_ari", "spectral_ari"]): 
        sub_df = df[df["vary"] == vary]
        sub_df = sub_df.groupby([f"{vary}_plus", f"{vary}_minus"])[metric].mean().reset_index()
        matrix = sub_df.pivot(index=f"{vary}_plus", columns=f"{vary}_minus", values=metric)
        
        ax = axarr[j, i]
        sns.heatmap(matrix, ax=ax, annot=False, fmt=".2f", cmap="YlGnBu", xticklabels=2, yticklabels=2)
        ax.set_title(f"{metric} vs {vary}")
        ax.set_xlabel(fr"$\{vary}_-$")
        ax.set_ylabel(fr"$\{vary}_+$")
        ax.set_aspect("equal")
        
        # make the x and y ticks only go to 2 decimal places
        # ax.set_xticklabels([f"{x:.2f}" for x in ax.get_xticklabels()])
        # labels = [f'{tick:0<5.1f}'[:-1].rstrip('.') for tick in ax.get_xticks()]

        # ax.set_xticklabels(labels)
        # print(ax.get_yticks())


        # ax.set_xticklabels([f"{x:.2f}" for x in ax.get_xticks()])
        # ax.set_yticklabels([f"{y:.2f}" for y in ax.get_yticks()])

plt.tight_layout()
plt.savefig("fig/community-synthetic-data-viz.png", dpi=300, bbox_inches="tight")
