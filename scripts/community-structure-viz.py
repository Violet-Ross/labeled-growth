from src.poisson_hypergraph import GH
from matplotlib import pyplot as plt
import pandas as pd
import os 
import seaborn as sns

import xgi
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from src.algorithms.sem import sem_functions
import pandas as pd
sem = sem_functions()


sns.set_style("whitegrid")

df = pd.DataFrame()

for filename in os.listdir("throughput/community-structure-sims"):
    if filename.endswith(".csv"):
        df = pd.concat([df, pd.read_csv(f"throughput/community-structure-sims/{filename}")], ignore_index=True)
        


params = ["theta_1", "theta_2", "theta_3", "theta_4", "theta_5", "theta_6"]

agged = df.groupby(params).agg({"modularity" : "mean", "proportion_pure_edges" : "mean", "mean_majority_proportion" : "mean"}).reset_index()


col_to_plot = "modularity"

agged["theta_2"].unique()

fig, axarr = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

for i in range(2): 
    theta_4 = agged["theta_4"].unique()[i]
    sub = agged[agged["theta_4"] == theta_4]
    sub["diff"] = sub["theta_1"] - sub["theta_2"]
    
    
    sns.lineplot(data=sub, x="diff", y=col_to_plot, hue="theta_3", palette="viridis", ax=axarr[i], alpha = 0.5, legend = i == 0, linewidth = 2.5)
    axarr[i].set_title(f"theta_4 = {theta_4}")
    axarr[i].set_ylim(0, None)

plt.savefig("fig/community-structure-sims-line.png", dpi=300, bbox_inches="tight")




# COMBINED FIG 


def clique_projection_modularity(GH): 
    """
    unweighted so possibly not that reliable, would want to use weighted graph instead but not implemented in XGI. Not a hard implementation but likely slow. 
    """
    
    A = xgi.linalg.hypergraph_matrix.adjacency_matrix(GH.H, weighted = True)
    z = np.array(GH.get_labels())
    k = A.sum(axis=1)
    m = A.sum() / 2
    Delta = z[:, None] == z[None, :] 
    return 1/(2*m)*((A - np.outer(k, k) / (2 * m))*Delta).sum()


fig, ax = plt.subplots(1, 1, figsize=(4, 3))

theta_4 = 0.5
sub = agged[agged["theta_4"] == theta_4]

palette = sns.color_palette("viridis", n_colors=sub["theta_3"].nunique())

sns.lineplot(data=sub, x="theta_1", y=col_to_plot, hue="theta_3", palette=palette, ax=ax, alpha = 0.8, legend = True, linewidth = 2.5, zorder = 100)
ax.set_ylim(0, None)
ax.set_xlabel(r"$\eta_+ = 1 - \eta_-$")
ax.set_ylabel(r"$Q$")
# change the legend title 
# reverse the order of the legend items 

handles, labels = ax.get_legend_handles_labels()
handles, labels = reversed(handles), reversed(labels)

new_labels = [f"$\\lambda_+$ = {label}" for label in labels]
ax.legend(handles, 
          new_labels, 
          title = None)



inset_1 = ax.inset_axes(
   [1.0, 0.5, 0.5, 0.5]
)

inset_2 = ax.inset_axes(
   [1.0, 0.0, 0.5, 0.5]
)


inset_3 = ax.inset_axes(
   [-0.6, 0.5, 0.5, 0.5]
)

inset_4 = ax.inset_axes(
   [-0.6, 0.0, 0.5, 0.5]
)
# we want to visualize a small hypergraph with two parameter sets corresponding to parts of the plot here. 

n_steps = int(30)

choices_to_highlight = [
    {"theta_1" : 0.95, "theta_3" : 2.0},
    {"theta_1" : 0.95, "theta_3" : 0.5},
    {"theta_1" : 0.55, "theta_3" : 2.0},
    {"theta_1" : 0.55, "theta_3" : 0.5},
]


for i, choice in enumerate(choices_to_highlight):
    
   point = sub[(sub["theta_1"] == choice["theta_1"]) & (sub["theta_3"] == choice["theta_3"])]
   # hollow square marker
   ax.plot(point["theta_1"], point[col_to_plot], marker = "s", color = "black", markersize = 8, fillstyle = 'none', zorder = 1000)
   
   
   eta_plus = choice["theta_1"]
   eta_minus = 1 - eta_plus
   lambda_plus = choice["theta_3"]
   lambda_minus = 0.5
   gamma_plus = 0.2
   gamma_minus = 0.1 
   

   inset_ax = [inset_1, inset_2, inset_3, inset_4][i]
    
    
   expected_modularity = point[col_to_plot].values[0]
   actual_modularity = -10
   
   while not np.isclose(expected_modularity, actual_modularity, atol = 0.001):
    
      H = xgi.Hypergraph([[0, 1, 2], [3, 4, 5]])
      H.set_node_attributes({0 : 0, 1 : 0, 2 : 0, 3 : 1, 4 : 1, 5 : 1}, name = "label")
      growing_hypergraph = GH(H, [0, 1], eta_plus, eta_minus)
      growing_hypergraph.add_hyperedge(n_steps, gamma_plus, gamma_minus, lambda_plus, lambda_minus)
      actual_modularity = clique_projection_modularity(growing_hypergraph)
      print(f"Expected modularity: {expected_modularity}, actual modularity: {actual_modularity}") 
    
   labels = growing_hypergraph.get_labels()
   H = growing_hypergraph.H
   
   
   H.cleanup(isolates = True)
   H.set_edge_attributes({e : H.edges.size(e) for e in H.edges}, name = "size")
   
   H = growing_hypergraph.H
   pos = xgi.barycenter_spring_layout(H, seed=1)
   pos = {i : 0.1*pos[i] for i in H.nodes}
   
   node_border_colors = ["black" if labels[i] == 0 else "white" for i in H.nodes]        

   xgi.draw(H, 
         pos = pos,
         node_fc=labels, 
         ax=inset_ax,
         node_fc_cmap=plt.cm.gray_r, 
         edge_fc_cmap=plt.cm.inferno,
         edge_vmin=0, edge_vmax=10,
         edge_lw = 0,
         node_ec = node_border_colors,
         node_lw = .3, 
         node_size = 5,
         hull = True, 
         radius = 0.005, 
         alpha = 0.2,
         )

plt.savefig("fig/community-structure-sims-combined.png", dpi=300, bbox_inches="tight", pad_inches=0.0)

    
