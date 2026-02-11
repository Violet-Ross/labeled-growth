import xgi
import numpy as np

# read in data from csv_to_xgi.py
H_json = xgi.read_json("gender_coauth.json")

nodes = np.array(list(H_json.nodes))
labels = np.array(H_json.nodes.attrs('label').aslist())

# remove nodes without gender label
H_json.remove_nodes_from(nodes[labels == '-'])

# record M as 0 and F as 1 for all nodes
genders = H_json.nodes.attrs('label').asdict()
labels = []
for party in list(genders.values()):
    if party == 'M':
        labels.append(0)
    if party == 'F':
        labels.append(1)

new_nodes = list(range(len(list(H_json.nodes))))
old_to_new_dic = dict(zip(list(H_json.nodes), new_nodes))
new_edges = [{old_to_new_dic.get(node) for node in edge} for edge in H_json.edges.members()]

# create new dict using our binary labels
label_dict = dict(zip(new_nodes, labels))

# make new hypergraph
new_H = xgi.Hypergraph(new_edges)
new_H.set_node_attributes(label_dict, name = "label")

# save the hypergraph locally
xgi.write_json(new_H, "gender_coauth_sorted.json")
