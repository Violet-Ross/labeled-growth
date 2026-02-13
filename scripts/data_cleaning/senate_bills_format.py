import xgi
import numpy as np

H = xgi.load_xgi_data("senate-bills")

party_affs = H.nodes.attrs('affiliation').asdict() # get party affiliation for nodes
new_nodes = sorted([int(node) - 1 for node in H.nodes]) # label nodes in increasing order, index from 0
new_edges = [{int(node) - 1 for node in edge} for edge in H.edges.members()] # describe edges in terms of new node labels

# record dem as 0 and rep as 1 for all nodes
labels = []
for party in list(party_affs.values()):
    if party == 'Democrat':
        labels.append(0)
    if party == 'Republican':
        labels.append(1)

# create new dict using our binary labels
label_dict = dict(zip(new_nodes, labels))

# make new hypergraph
new_H = xgi.Hypergraph(new_edges)
new_H.set_node_attributes(label_dict, name = "label")

# save the hypergraph locally
xgi.write_json(new_H, "throughput/senate_bills.json")