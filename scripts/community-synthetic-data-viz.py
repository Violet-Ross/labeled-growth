import sys
import os
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import scripts.figure_settings as fs

fs.set_fonts()

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
        value_vars=["simulated_annealing_ari", "spectral_ari"],
        var_name="metric", value_name="value")

# remove rows where the condition is "guess" and the metric is "spectral_ari"
# df = df[~((df["metric"] == "spectral_ari") & (df["condition"] == "guess"))]

df.groupby(["vary", "metric", "condition"])["value"].size().reset_index()

df["metric_type"] = df["metric"] + "_" + df["condition"]

df = df[df["metric_type"] != "spectral_ari_nishimori"]

sns.set_style("whitegrid")

# new attempt

titles = {
    "simulated_annealing_ari_guess": "Simulated annealing\n(fixed)", 
    "simulated_annealing_ari_nishimori": "Simulated annealing\n(Nishimori)", 
    "spectral_ari_guess": "Spectral\nclustering", 
}

fig, axarr = plt.subplots(1, 3, figsize=(11, 3), sharey=True)

color_palette = [fs.palette[0], fs.lighten(fs.palette[0]),  "darkgrey"]
markers = ["o", "s", "^"]
linestyles = ["-", "--", "-"]
labs = ["(a)", "(b)", "(c)"]
# for i, vary in enumerate(["eta", "lambda", "gamma"]): 
for i, vary in enumerate(["eta", "lambda", "gamma"]):
    for j, metric in enumerate(titles.keys()):
        sub = df[df["vary"] == vary]
        sub = sub[sub["metric_type"] == metric]
        sub.drop(columns=["metric_type"], inplace=True)
        sub = sub.sort_values(by=[f"{vary}_plus"])
        sub = sub.groupby([f"{vary}_plus"])["value"].mean().reset_index()
        
        axarr[i].plot(sub[f"{vary}_plus"], sub["value"], label=titles[metric], color=color_palette[j], linewidth=2, linestyle=linestyles[j])
        axarr[i].scatter(sub[f"{vary}_plus"], sub["value"], color=color_palette[j], s=40, label = titles[metric], marker=markers[j])
        axarr[i].set_xlabel(fr"$\{vary}_+$", fontsize=14)
    
    axarr[i].annotate(labs[i], xy=(0.03, 0.87), xycoords='axes fraction', fontsize=20, fontweight='bold', bbox=dict(facecolor='white', edgecolor='white', alpha=0.5))

# axarr[i].set_ylim(0, None)
        
handler, labeler = axarr[0].get_legend_handles_labels()
handler = [(handler[0],handler[1]),
       (handler[2],handler[3]), 
       (handler[4],handler[5])]
labeler = labeler[::2]

plt.legend(handler, labeler, loc="lower center", bbox_to_anchor=(-0.7, -0.6), ncol=3, frameon=False, handletextpad=0.5, columnspacing=1.5, fontsize=14)


# axarr[2].legend(handler, labeler, loc="center left", bbox_to_anchor=(1, 0.5))



axarr[0].set_ylabel("ARI", fontsize=14)
plt.tight_layout()

plt.savefig("fig/community-synthetic-data-viz-line.png", dpi=300, bbox_inches="tight")
        










# # selection for main text

# fig, axarr = plt.subplots(1, 3, figsize=(11, 4))
# cbar_ax = fig.add_axes([.91, .15, .02, .7])



# vary = "eta"
# for j, metric in enumerate(titles.keys()):
#     sub = df[df["vary"] == vary]
#     sub = sub[sub["metric_type"] == metric]
#     sub.drop(columns=["metric_type"], inplace=True)
#     sub = sub.groupby([f"{vary}_plus", f"{vary}_minus"])["value"].mean().reset_index()
#     matrix = sub.pivot(index=f"{vary}_plus", columns=f"{vary}_minus", values="value")

#     ax = axarr[j]
#     im = sns.heatmap(matrix, ax=ax, annot=False, fmt=".2f", cmap="inferno", xticklabels=2, yticklabels=2, vmax=0.6, vmin = 0, cbar = j == 2, cbar_ax=cbar_ax if j == 2 else None)
#     ax.set_title(titles[metric])
    
#     if j == 0:
#         ax.set_ylabel(fr"$\{vary}_+$")
#     else: 
#         ax.set_ylabel("")
#     ax.set_xlabel(fr"$\{vary}_-$")
#     # ax.set_ylabel(fr"$\{vary}_+$")
#     ax.set_aspect("equal")

#     # reverse the y axis
#     ax.invert_yaxis()

#     # make the labels on the y axis horizontal, no rotation
#     ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

# fig.tight_layout(rect=[0, 0, .9, 1])

# # plt.tight_layout()
# plt.savefig("fig/community-synthetic-data-viz-main.png", dpi=300, bbox_inches="tight")


# fig, axarr = plt.subplots(3, 3, figsize=(12, 12))
# cbar_ax = fig.add_axes([.91, .15, .02, .7])



# for i, vary in enumerate(["eta", "lambda", "gamma"]): 
#     for j, metric in enumerate(titles.keys()):
        
#         sub = df[df["vary"] == vary]
#         sub = sub[sub["metric_type"] == metric]
#         sub.drop(columns=["metric_type"], inplace=True)
        
        
#         sub = sub.groupby([f"{vary}_plus", f"{vary}_minus"])["value"].mean().reset_index()
#         matrix = sub.pivot(index=f"{vary}_plus", columns=f"{vary}_minus", values="value")
        
#         ax = axarr[j, i]
#         sns.heatmap(matrix, ax=ax, annot=False, fmt=".2f", cmap="inferno", xticklabels=2, yticklabels=2, vmax=0.75, vmin = 0, cbar = j == 2, cbar_ax=cbar_ax if j == 2 else None)
#         ax.set_title(titles[metric])
        
            
#         if i == 0:
#             ax.set_ylabel(fr"$\{vary}_+$")
#         else: 
#             ax.set_ylabel("")
        
#         if j == 2: 
#             ax.set_xlabel(fr"$\{vary}_-$")
#         else:
#             ax.set_xlabel("")

#         ax.set_aspect("equal")
        
#         # reverse the y axis
#         ax.invert_yaxis()
        
#         # make the labels on the y axis horizontal, no rotation
#         ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

# fig.tight_layout(rect=[0, 0, .9, 1])
# plt.savefig("fig/community-synthetic-data-viz-supplementary.png", dpi=300, bbox_inches="tight")
