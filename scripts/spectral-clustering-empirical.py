import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, project_root)

import argparse
import itertools
import pandas as pd
# in a .py file, just this import works
import xgi
from src.poisson_hypergraph import GH
from src.algorithms.simulated_annealing import SimulatedAnnealingApprox
import numpy as np
import networkx as nx
import csv
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import pickle as pkl
import pandas as pd 
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
from itertools import combinations

def spectral_clustering_comparison(g): 
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

    adjacency_matrix = nx.to_numpy_array(G, weight="weight")
    spectral = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=0)
    labels = spectral.fit_predict(adjacency_matrix)

    return labels

data_sets = {
    "senate_bills": "senate_bills",
    "highschool_class": "highschool",
    "highschool_gender": "highschool_gender",
    "primaryschool_gender": "primaryschool_gender",
    "primaryschool_class": "primaryschool",
    "gender_coauth": "gender_coauth_shrunk"
}

mode = "w"
header = True

if __name__ == "__main__":
    # figure out which data set we are using
    args = argparse.ArgumentParser(description='Run simulated annealing on a user-specified prepared data set.')
    args.add_argument('--data_set', type=str, choices=data_sets.keys(), help='The data set to run simulated annealing on. Must be one of: ' + ', '.join(data_sets.keys()))
    args.add_argument("--job_id", type=int, help="The job ID for this run, used to differentiate output files when running multiple jobs.")
    
    args = args.parse_args()
    
    data_set = args.data_set
    job_id = args.job_id

    results_folder = data_sets[data_set] # set default name of results folder to the title of dataset
    # if args.results_folder != None:
    #     results_folder = args.results_folder # override if provided

    path = f"throughput/spectral-clustering/{results_folder}"
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + "/labels", exist_ok=True)
    os.makedirs(path + "/metrics", exist_ok=True)
    
    # initialize 
    # H = pkl.load(open(f"throughput/{data_sets[data_set]}.pkl", "rb")) unsure of why this is used
    H = xgi.read_json(f"throughput/{data_sets[data_set]}.json", nodetype=int)
    g = GH(H, [0, 1], 0, 0)

    labels = spectral_clustering_comparison(g)
    ari = adjusted_rand_score(labels, g.get_labels())

    df = pd.DataFrame({
        "labels"   : [labels]
    })
                
    df.to_csv(f'{path}/labels/{job_id}.csv', index=False, mode = mode, header = header)
    
    df = pd.DataFrame({
        "ari"        : [ari]
    })
    
    df.to_csv(f'{path}/metrics/{job_id}.csv', index=False, mode = mode, header = header)
    
    mode = "a"
    header = False  