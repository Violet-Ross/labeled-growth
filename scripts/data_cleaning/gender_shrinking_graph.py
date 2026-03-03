import xgi
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

gender = xgi.read_json("throughput/gender_coauth_sorted.json", nodetype=int, edgetype = int)
senate = xgi.read_json("throughput/senate_bills.json", nodetype=int)

target_num_edges = senate.num_edges

# Define functions
def hyperedge_ego(H, e):

    # nodes in focal hyperedge
    ego_nodes = set(H.edges.members(e))

    # hyperedges intersecting it
    ego_edges = [
        eid for eid in H.edges
        if ego_nodes & set(H.edges.members(eid))
    ]

    # nodes appearing in those hyperedges
    nodes_in_ego = set()

    for eid in ego_edges:
        nodes_in_ego.update(H.edges.members(eid))

    # induce on BOTH edges and nodes
    return xgi.subhypergraph(H, nodes=nodes_in_ego, edges=ego_edges)

def hypergraph_union(H1, H2):
    H = H1.copy()

    # add nodes + attributes
    for n, attrs in H2.nodes.items():
        H.add_node(n, **attrs)

    # add edges + attributes
    for e, attrs in H2.edges.items():
        H.add_edge(H2.edges.members(e), **attrs)

    return H

def shrink_hypergraph(H, num_pivots):
    pivot_edges = np.random.randint(0, H.num_edges, size = num_pivots)
    ego_union = xgi.Hypergraph()
    for hyperedge in pivot_edges:
        ego_net = hyperedge_ego(H, hyperedge)
        ego_union = hypergraph_union(ego_union, ego_net)
    return ego_union

# Estimate number of pivot edges needed to generate a subgraph with
# the same number of hyperedges as senate bills data
edge_counts = []
for size in range(1, 20):
    H = shrink_hypergraph(gender, size)
    edge_counts.append(H.num_edges)

# Predict the required number of pivot edges 
x = np.array(range(1, 20))
y = np.array(edge_counts)
m, b = np.polyfit(x, y, 1)
x_pred = (target_num_edges - b) / m

start = time.time()
H = shrink_hypergraph(gender, np.ceil(x_pred).astype(int))
end = time.time()
xgi.write_json(H, "throughput/gender_coauth_shrunk.json")
total_time = end - start
print("Subgraph generated! It took ", int(total_time // 60), "minutes and", int(total_time % 60), "seconds")