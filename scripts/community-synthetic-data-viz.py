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


    
# round all the parameters to 2 decimal places
for col in ["eta_plus", "eta_minus", "lambda_plus", "lambda_minus", "gamma_plus", "gamma_minus"]:
    df[col] = df[col].round(2)

df = df.melt(id_vars=["eta_plus", "eta_minus", "lambda_plus", "lambda_minus", "gamma_plus", "gamma_minus", "condition", "vary"], 
        value_vars=["simulated_annealing_ari", "spectral_ari", "gradient_descent_ari"],
        var_name="metric", value_name="value")

# remove rows where the condition is "guess" and the metric is "spectral_ari"
# df = df[~((df["metric"] == "spectral_ari") & (df["condition"] == "guess"))]

df.groupby(["vary", "metric", "condition"])["value"].size().reset_index()

df["metric_type"] = df["metric"] + "_" + df["condition"]

df = df[df["metric_type"] != "spectral_ari_nishimori"]

sns.set_style("whitegrid")
fig, axarr = plt.subplots(5, 3, figsize=(15, 13))

titles = {
    "simulated_annealing_ari_guess": "Simulated annealing (fixed)", 
    "simulated_annealing_ari_nishimori": "Simulated annealing (Nishimori)", 
    "spectral_ari_guess": "Spectral clustering", 
    "spectral_ari_nishimori": "Spectral clustering",
    "gradient_descent_ari_guess": "Gradient descent (fixed)",
    "gradient_descent_ari_nishimori": "Gradient descent (Nishimori)"
}


for i, vary in enumerate(["eta", "lambda", "gamma"]): 
    for j, metric in enumerate(df["metric_type"].unique()):
        
        sub = df[df["vary"] == vary]
        sub = sub[sub["metric_type"] == metric]
        sub.drop(columns=["metric_type"], inplace=True)
        
        
        sub = sub.groupby([f"{vary}_plus", f"{vary}_minus"])["value"].mean().reset_index()
        matrix = sub.pivot(index=f"{vary}_plus", columns=f"{vary}_minus", values="value")
        
        ax = axarr[j, i]
        sns.heatmap(matrix, ax=ax, annot=False, fmt=".2f", cmap="inferno", xticklabels=2, yticklabels=2, vmax=0.8, vmin = 0)
        ax.set_title(titles[metric])
        ax.set_xlabel(fr"$\{vary}_-$")
        ax.set_ylabel(fr"$\{vary}_+$")
        ax.set_aspect("equal")
        
        # reverse the y axis
        ax.invert_yaxis()
        
        # make the labels on the y axis horizontal, no rotation
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig("fig/community-synthetic-data-viz.png", dpi=300, bbox_inches="tight")
