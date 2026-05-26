import xgi
import numpy as np

# Step 1: Load
H_json = xgi.read_json("throughput/gender_coauth.json")

# Step 2: Remove nodes with missing gender label
nodes = np.array(list(H_json.nodes))
labels = np.array(H_json.nodes.attrs('label').aslist())
H_json.remove_nodes_from(nodes[labels == '-'])

# Step 3: Build mapping AFTER removal so it only contains surviving nodes
surviving_nodes = list(H_json.nodes)
new_nodes = list(range(len(surviving_nodes)))
old_to_new_dic = dict(zip(surviving_nodes, new_nodes))

# Step 4 & 5: Build new hypergraph, filtering removed nodes from edge members
new_H = xgi.Hypergraph()

for node in new_nodes:
    new_H.add_node(node)

for old_edge_id in H_json.edges:
    old_members = H_json.edges.members(old_edge_id)
    new_members = {old_to_new_dic[node] for node in old_members if node in old_to_new_dic}

    if not new_members:
        continue

    attrs = H_json.edges[old_edge_id]
    new_H.add_edge(new_members, **attrs)

# Step 6: Set binary node labels (M=0, F=1)
genders = H_json.nodes.attrs('label').asdict()
binary_labels = {old_to_new_dic[node]: (0 if gender == 'M' else 1)
                 for node, gender in genders.items()}
new_H.set_node_attributes(binary_labels, name="label")

# Step 7: Sort edges by year, assign integer names in that order
sorted_edge_ids = sorted(new_H.edges, key=lambda e: new_H.edges[e].get("year", 0))

H_sorted = xgi.Hypergraph()

for node in new_H.nodes:
    H_sorted.add_node(node, **new_H.nodes[node])

for i, old_edge_id in enumerate(sorted_edge_ids):
    members = new_H.edges.members(old_edge_id)
    attrs = new_H.edges[old_edge_id]
    H_sorted.add_edge(members, id=i, **attrs)

# Step 8: Save
xgi.write_json(H_sorted, "throughput/gender_coauth_sorted.json")