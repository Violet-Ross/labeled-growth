import xgi
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import time

gender_shrunk = xgi.read_json("throughput/gender_coauth_shrunk.json", nodetype=int, edgetype = int)
gender = xgi.read_json("throughput/gender_coauth_sorted.json", nodetype=int, edgetype = int)
senate= xgi.read_json("throughput/senate_bills.json", nodetype=int, edgetype = int)

# Compare shrunk coauth network to senate bills network
edge_ratio_senate = gender_shrunk.num_edges / senate.num_edges
print("Edge count of the shrunk coauth network is equal to the senate edge count times", edge_ratio_senate)

node_ratio_senate = gender_shrunk.num_nodes / senate.num_nodes
print("Node count of the shrunk coauth network is equal to the senate node count times", node_ratio_senate)

# Compare shrunk coauth network to original coauth network
edge_ratio_gender = gender_shrunk.num_edges / gender.num_edges
print("Edge count reduced to", edge_ratio_gender, "of its original size. Total number of edges is now", gender_shrunk.num_edges)

node_ratio_gender = gender_shrunk.num_nodes / gender.num_nodes
print("Node count reduced to", node_ratio_gender, "of its original size. Total number of nodes is now", gender_shrunk.num_nodes)

prop_women = sum(gender.nodes.attrs("label").aslist()) / len(gender.nodes.attrs("label").aslist()) 
prop_women_shrunk = sum(gender_shrunk.nodes.attrs("label").aslist()) / len(gender_shrunk.nodes.attrs("label").aslist()) 
print("Proportion of women in the original coauth network is", prop_women, ". In the shrunk coauth network it is", prop_women_shrunk)

centers, heights = xgi.degree_histogram(gender)
list_of_edges_sizes = gender.edges.size.aslist()
size_counts = dict(Counter(list_of_edges_sizes))

centers_shrunk, heights_shrunk = xgi.degree_histogram(gender_shrunk)
list_of_edges_sizes_shrunk = gender_shrunk.edges.size.aslist()
size_counts_shrunk = dict(Counter(list_of_edges_sizes_shrunk))

sns.set_style("whitegrid")
sns.set_palette("Dark2")

plt.rcParams.update({'font.size': 20})
plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15) 

fig, axs = plt.subplots(1, 2, sharey=True)
fig.set_figwidth(20)
fig.set_figheight(10)
axs[0].scatter(centers, np.array(heights) / len(gender.nodes), label = "original")
axs[0].scatter(centers_shrunk, np.array(heights_shrunk) / len(gender_shrunk.nodes), label = "shrunk")
axs[0].plot(range(1, 10**3), [x ** (-2) for x in range(1, 10**3)], c = "black")
axs[0].set_ylabel("P(k)")
axs[0].set_xlabel("k")
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].legend(loc='center left', bbox_to_anchor=(-0.6, 0.5))

axs[1].scatter(list(size_counts.keys()), np.array(list(size_counts.values())) / len(gender.nodes))
axs[1].scatter(list(size_counts_shrunk.keys()), np.array(list(size_counts_shrunk.values())) / len(gender_shrunk.nodes))
axs[1].set_ylabel("P(s)")
axs[1].set_xlabel("s")
axs[1].set_yscale('log')

fig.savefig('fig/shrunk_vs_og_degree_dist.png', dpi=300, bbox_inches = "tight")