import xgi
import pandas as pd
import numpy as np
import networkx as nx

# read in csv files
contacts = np.array(pd.read_csv('data/highschool/HighSchool2013_proximity_net.csv', header=None))
genders = np.array(pd.read_csv('data/highschool/metadata.txt', header=None, sep="\t"))

hyperedges = []
prev_added_hyperedges = set()
class_labels = {}
g = nx.Graph()

prev_timestep = None
for contact in contacts:
    # get each line of csv, split into attributes
    timestep, index0, index1, class0, class1 = contact[0].split(" ")

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

        # Depracated
        # for hyperedge in nx.connected_components(g):
        #     if hyperedge not in prev_added_hyperedges:
        #         hyperedges.append(hyperedge)
        #     prev_added_hyperedges.append
        #     # print("hyperedge: " + str(hyperedge))
        
        # clear graph
        g = nx.Graph()
    g.add_edge(index0, index1)


# create hypergraph from the edgelist
H = xgi.Hypergraph(hyperedges)

H.cleanup(singletons=False, multiedges=True, relabel=False, isolates=True)


# make 9 classes into binary labeled
# for node, class_label in class_labels.items():
#     if class_label == "2BIO1" or class_label == "2BIO2" or class_label == "2BIO3":
#         class_labels[node] = 1
#     else:
#         class_labels[node] = 0
for node, class_label in class_labels.items():
    if class_label == "2BIO1":
        class_labels[node] = 0
    elif class_label == "2BIO2":
        class_labels[node] = 1
    elif class_label == "2BIO3":
        class_labels[node] = 2
    elif class_label == "MP":
        class_labels[node] = 3
    elif class_label == "MP*1":
        class_labels[node] = 4
    elif class_label == "MP*2":
        class_labels[node] = 5
    elif class_label == "PC":
        class_labels[node] = 6
    elif class_label == "PC*":
        class_labels[node] = 7
    elif class_label == "PSI*":
        class_labels[node] = 8
    else:
        print("ERROR")

# remake to retain orginal labels!

# set the node labels
H.set_node_attributes(class_labels, name = "label")

# want to index the nodes from 0 to n, the number of nodes

# store old id
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
xgi.write_json(H2, "throughput/highschool.json")


# make another file with gender labels instead of classes
H = xgi.Hypergraph(hyperedges)

gender_dict = {}

for gender in genders:
    gender_dict[str(gender[0])] = gender[2]

# make 9 classes into binary labeled
for node, gender_label in gender_dict.items():
    if gender_label == "M":
        gender_dict[node] = 1
    elif gender_label == "F":
        gender_dict[node] = 0
    else:
        print(node)
        H.remove_node(node)

# remove singleton edges
H.cleanup(singletons=False, multiedges=True, relabel=False, isolates=True)

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
xgi.write_json(H2, "throughput/highschool_gender.json")

