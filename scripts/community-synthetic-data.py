import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, project_root)

import argparse
import itertools
import pandas as pd
# in a .py file, just this import works
from src.poisson_hypergraph import GH # custom hypergraph class
from src.algorithms.simulated_annealing import SimulatedAnnealingApprox
import xgi
import csv

from sklearn.metrics import adjusted_rand_score 
import networkx as nx
import numpy as np
from itertools import combinations

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering
sns.set_style("whitegrid")

def initial_condition_GH(theta, timesteps):
    eta_plus, eta_minus, gamma_nu, gamma_nr, gamma_eu, gamma_er = theta
    H = xgi.Hypergraph([[0,1,2, 3], [3, 4, 5]])
    H.set_node_attributes({0:0,1:0,2:0,3:0,4:0,5:1}, name="label")
    g = GH(H,[0, 1], eta_plus , eta_minus)
    g.add_hyperedge(timesteps, gamma_nu , gamma_nr , gamma_eu , gamma_er)
    return g

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

def simulated_annealing(g, guessed_theta):
    sa = SimulatedAnnealingApprox(g, guessed_theta, approx=100, novel=True)
    algo_timesteps = len(g.nodes)*20

    best_l = -float('inf')
    labels_at_max_ll = None
    for step_num in range(algo_timesteps):
        # print(best_l)
        sa.step()
        # print(sa.likelihoods_per_step[-1])
        if sa.likelihoods_per_step[-1] > best_l:
            best_l = sa.likelihoods_per_step[-1]
            labels_at_max_ll = sa.labels
            
    return labels_at_max_ll

def run_experiment(ETA_PLUS, ETA_MINUS, LAMBDA_PLUS, LAMBDA_MINUS, GAMMA_PLUS, GAMMA_MINUS, timesteps, fpath, guessed_theta = None, **kwargs):
    
    THETA = itertools.product(ETA_PLUS, ETA_MINUS, LAMBDA_PLUS, LAMBDA_MINUS, GAMMA_PLUS, GAMMA_MINUS)
    
    fpath_exists = os.path.exists(fpath)
    mode = "a" if fpath_exists else "w"
    header = not fpath_exists
    
    
    for theta in THETA:
        g = initial_condition_GH(theta, timesteps)
    
    # spectral clustering comparison
        spectral_labels = spectral_clustering_comparison(g)
        spectral_ari    = adjusted_rand_score(g.get_labels(), spectral_labels)
        
        if guessed_theta is None:
            guessed_theta = theta
        simulated_annealing_labels = simulated_annealing(g, guessed_theta)
        
        
        simulated_annealing_ari = adjusted_rand_score(g.get_labels(), simulated_annealing_labels)
        
        df = pd.DataFrame(
            {
                "eta_plus" : theta[0],
                "eta_minus" : theta[1],
                "lambda_plus" : theta[2],
                "lambda_minus" : theta[3],
                "gamma_plus" : theta[4],
                "gamma_minus" : theta[5],
                "spectral_ari" : spectral_ari, 
                "simulated_annealing_ari" : simulated_annealing_ari
            }, index = [0]
        )
        
        for key, value in kwargs.items():
            df[key] = value
        
        df.to_csv(fpath, index=False, mode = mode, header = header)
    
if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--steps", type=int, default=10)
    args.add_argument("--resolution", type=int, default=11)
    args.add_argument("--runs", type=int, default=1)
    args.add_argument("--job", type=int, default=0)
    args = args.parse_args()
    timesteps = args.steps
    resolution = args.resolution
    runs = args.runs
    job = args.job
    
    # create directory throughput/community/synthetic if it doesn't exist
    # first, delete it even if it is not empty
    path = "throughput/community/synthetic"
    if os.path.exists(path):
        for filename in os.listdir(path):
            os.remove(os.path.join(path, filename))
        # os.rmdir(path)
    os.makedirs(path, exist_ok=True)
    
    # first suite: vary eta_plus and eta_minus, hold lambda_plus and lambda_minus fixed
    
    
    for run in range(runs):
        
        # vary copy parameter eta
        ETA_PLUS = np.linspace(0.5, .95, resolution)
        ETA_MINUS = np.linspace(0.05, 0.5, resolution)
        
        LAMBDA_PLUS = [1.5]
        LAMBDA_MINUS = [0.5]
        
        GAMMA_PLUS = [0.1]
        GAMMA_MINUS = [0.1]
        
        run_experiment(ETA_PLUS, ETA_MINUS, LAMBDA_PLUS, LAMBDA_MINUS, GAMMA_PLUS, GAMMA_MINUS, timesteps, f"throughput/community/synthetic/{job}.csv", guessed_theta = None, run = run, vary = "eta")
        
        # vary external parameter lambda
        ETA_PLUS = [0.9]
        ETA_MINUS = [0.1]
        LAMBDA_PLUS = np.linspace(1, 2, resolution+1)
        LAMBDA_MINUS = np.linspace(0.01, 1, resolution+1)
        GAMMA_PLUS = [0.1]
        GAMMA_MINUS = [0.1]
        
        run_experiment(ETA_PLUS, ETA_MINUS, LAMBDA_PLUS, LAMBDA_MINUS, GAMMA_PLUS, GAMMA_MINUS, timesteps, f"throughput/community/synthetic/{job}.csv", guessed_theta = None, run = run, vary = "lambda")
        
        # vary novel parameter gamma
        ETA_PLUS = [0.9]
        ETA_MINUS = [0.1]
        LAMBDA_PLUS = [1.5]
        LAMBDA_MINUS = [0.5]
        GAMMA_PLUS = np.linspace(0.05, 0.5, resolution)
        GAMMA_MINUS = np.linspace(0.05, 0.5, resolution)
        run_experiment(ETA_PLUS, ETA_MINUS, LAMBDA_PLUS, LAMBDA_MINUS, GAMMA_PLUS, GAMMA_MINUS, timesteps, f"throughput/community/synthetic/{job}.csv", guessed_theta = None, run = run, vary = "gamma")
        
    
    
    
