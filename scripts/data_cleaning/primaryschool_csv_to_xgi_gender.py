import xgi
import pandas as pd
import numpy as np
import networkx as nx

# read in csv files
contacts = np.array(pd.read_csv('data/primaryschool/primaryschool.csv', header=None, sep="\t"))
genders = np.array(pd.read_csv('data/primaryschool/metadata.txt', header=None, sep="\t"))

hyperedges = []
class_labels = {}
g = nx.Graph()
prev_added_hyperedges = set()

gender_dict = {}

for gender in genders:
    if gender[2] != "Unknown":
        gender_dict[(gender[0])] = gender[2]


prev_timestep = None
for contact in contacts:
    # get each line of csv, split into attributes
    timestep, index0, index1, class0, class1 = contact

    # add class labels to nodes
    class_labels[(index0)] = class0
    class_labels[(index1)] = class1

    if prev_timestep == None:
        prev_timestep = timestep

    # want interaction hyperedge where all people interacting in one timestep are in a hyperedge... 
    # can be multiple, so can't add all to hyperedge

    if prev_timestep != timestep:
        # print("timestep: " + str(prev_timestep))
        prev_timestep = timestep
        connected_components = set([tuple(hyperedge) for hyperedge in nx.connected_components(g)])

        hyperedges.extend(list(connected_components - prev_added_hyperedges))
        prev_added_hyperedges = connected_components
        
        # for hyperedge in nx.connected_components(g):
        #     hyperedges.append(hyperedge)
        #     # print("hyperedge: " + str(hyperedge))
        
        # clear graph
        g = nx.Graph()
    
    # don't add unknown gender nodes or teacher nodes (which all have unknown gender)
    if index0 in gender_dict and index1 in gender_dict:
        g.add_edge(index0, index1)



# create hypergraph from the edgelist
H = xgi.Hypergraph(hyperedges)

# remove singleton
H.cleanup(singletons=False, multiedges=True, relabel=False, isolates=True)


# make 9 classes into binary labeled
for node, gender_label in gender_dict.items():
    if gender_label == "M":
        gender_dict[node] = 1
    elif gender_label == "F":
        gender_dict[node] = 0
    else:
        print(node)
        H.remove_node(node)



# set the node labels
H.set_node_attributes(gender_dict, name = "label")

# save old labels
old_labels = {n: n for n in H.nodes}
H.set_node_attributes(old_labels, name="dataset_id")

# make new hypergraph with correct indexing
H2 = xgi.Hypergraph()
mapping = {old: i for i, old in enumerate(H.nodes)}

# Add nodes with attributes
for old_node in H.nodes:
    new_node = mapping[old_node]
    H2.add_node(new_node, **H.nodes[old_node])

# add edges
for edge in H.edges:
    new_edge = [mapping[n] for n in H.edges.members(edge)]
    H2.add_edge(new_edge, **H.edges[edge])

# save the hypergraph locally
xgi.write_json(H2, "throughput/primaryschool_gender.json")

