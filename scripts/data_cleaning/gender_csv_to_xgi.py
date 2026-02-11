import xgi
import pandas as pd
import numpy as np

# read in csv files
authors = np.array(pd.read_csv('authors.csv'))
general = np.array(pd.read_csv('general.csv'))

# generate list of edges (where an edge is a paper and the nodes within it are its author)
paper_ids = np.unique(authors[:,1])
paper_authors = []
paper_genders = []

for pid in paper_ids:
    pid_authors = authors[authors[:,1] == pid, 3]
    paper_authors.append(pid_authors)

paper_authors = np.array(paper_authors, dtype=object)
paper_authors = [paper.tolist() for paper in paper_authors]

# create hypergraph from the edgelist
H = xgi.Hypergraph(paper_authors)

# set the node labels (author gender)
author_gender = dict(zip(authors[:,3], authors[:,4]))
H.set_node_attributes(author_gender, name = "label")

# set the edge labels (paper id and year)
paper_id_dict = dict(zip(H.edges, paper_ids))
H.set_edge_attributes(paper_id_dict, name = "id")

id_year = dict(zip(general[:, 0], general[:, 1]))
edge_ids = H.edges.attrs("id").aslist()
years = [id_year.get(eid) for eid in edge_ids]

year_id_dict = dict(zip(H.edges, years))
H.set_edge_attributes(year_id_dict, name = "year")

# save the hypergraph locally
xgi.write_json(H, "gender_coauth.json")
