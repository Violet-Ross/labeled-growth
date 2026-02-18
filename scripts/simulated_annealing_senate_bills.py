import xgi
from src.poisson_hypergraph import GH
from src.algorithms.simulated_annealing import SimulatedAnnealingApprox
import numpy as np
import networkx as nx
import sys
import csv
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

H = xgi.load_xgi_data("senate-bills")

# Violet's Code Congress Sem
party_affs = H.nodes.attrs('affiliation').asdict()
new_nodes = sorted([int(node) - 1 for node in H.nodes])
new_edges = [{int(node) - 1 for node in edge} for edge in H.edges.members()]

# record dem as 0 and rep as 1 for all nodes
labels = []
for party in list(party_affs.values()):
    if party == 'Democrat':
        labels.append(0)
    if party == 'Republican':
        labels.append(1)

# create new dict using our binary labels
label_dict = dict(zip(new_nodes, labels))
sorted_label_dict = dict(sorted(label_dict.items()))

# make new hypergraph
new_H = xgi.Hypergraph(new_edges)
new_H.set_node_attributes(sorted_label_dict, name = "label")

# turn the data set into an object of the GH class (so we can perform SEM on it)
g = GH(new_H, [0, 1], 0, 0)

# # true_theta = [.43, .37, 0.001, .001, .91, .65] # Violet's approx
# true_thetas = [[.9, .1, .001, .001, 1, .25],
# [.8, .2, .5, .25, 1, .25],
# [.7, .3, .5, .25, 1, .25],
# [.99, .01, .5, .25, 1, .25],
# [.85, .15, 0.5, .25, .91, .65]]

true_theta = [.9, .1, .001, .001, 1, .25]
    
from itertools import combinations
def clique_projection_modularity_maximization_algo(g):
    H = g.H
    G = nx.Graph()
    G.add_nodes_from(H.nodes)

    # Clique projection
    for edge in H.edges.members():
        for u, v in combinations(edge, 2):
            if G.has_edge(u, v):
                G[u][v]["weight"] += 1
            else:
                G.add_edge(u, v, weight=1)

    partition = nx.community.greedy_modularity_communities(G, best_n=2)
    z = np.array([0 if node in partition[0] else 1 for node in G.nodes()])

    return z
    


# Simulated annealing run

sa = SimulatedAnnealingApprox(g, true_theta)

for step_num in range(len(g.nodes)*20):
    print("step num: " + str(step_num))

    sa.step()

    # with open('senate_bills_diff_params' + str(theta_index) + '.csv', 'a', newline="") as file:
    with open('./throughput/simulated_annealing_senate_bills.csv', 'a', newline="") as file:
        writer = csv.writer(file)
        writer.writerows([[step_num, sa.likelihoods_per_step[-1], sa.aris_per_step[-1]]])


# save labels
with open('./throughput/senate_bills_simulated_annealing.csv', 'a', newline="") as file:
    writer = csv.writer(file)
    writer.writerows([sa.labels])

    writer.writerows([[sa.max_LL, sa.max_LL_corresponding_ari]])
    writer.writerows([sa.max_LL_labels])


