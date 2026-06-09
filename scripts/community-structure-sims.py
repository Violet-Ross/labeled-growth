from src.poisson_hypergraph import GH
import xgi
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from src.algorithms.sem import sem_functions
import pandas as pd
sem = sem_functions()
from itertools import product
import argparse
import os 
import tqdm


# eta_plus = 0.9
# eta_minus = 0.1
# lambda_plus = 0.7
# lambda_minus = 0.1
# gamma_minus = 0.1
# gamma_plus = 0.25
# n_steps = 1000
# H = xgi.Hypergraph([[0, 1], [2, 3]])
# H.set_node_attributes({0 : 0, 1 : 0, 2 : 1, 3 : 1}, name = "label")
# growing_hypergraph = GH(H, [0, 1], eta_plus, eta_minus)
# growing_hypergraph.add_hyperedge(n_steps, lambda_plus, lambda_minus, gamma_plus, gamma_minus)

# A = xgi.linalg.hypergraph_matrix.adjacency_matrix(growing_hypergraph.H, weighted = True)
# A.todense().max()




def proportion_pure_edges(GH):
    H = GH.H
    labels = GH.get_labels()
    pure_edges = 0
    for e in H.edges.members():
        if len(set(labels[i] for i in e)) == 1:
            pure_edges += 1
    return pure_edges / len(H.edges)

def mean_majority_proportion(GH):
    H = GH.H
    labels = GH.get_labels()
    majority_proportions = []
    for e in H.edges.members():
        label_counts = {}
        for i in e:
            label_counts[labels[i]] = label_counts.get(labels[i], 0) + 1
        majority_count = max(label_counts.values())
        majority_proportions.append(majority_count / len(e))
    return np.mean(majority_proportions)

def clique_projection_modularity(GH): 
    """
    unweighted so possibly not that reliable, would want to use weighted graph instead but not implemented in XGI. Not a hard implementation but likely slow. 
    """
    
    A = xgi.linalg.hypergraph_matrix.adjacency_matrix(GH.H, weighted = True)
    z = np.array(GH.get_labels())
    k = A.sum(axis=1)
    m = A.sum() / 2
    Delta = z[:, None] == z[None, :] 
    return 1/(2*m)*((A - np.outer(k, k) / (2 * m))*Delta).sum()


if __name__ == "__main__":
    
    path = "throughput/community-structure-sims"
    
    # make directory if it doesn't exist already
    if not os.path.exists(path):
        os.makedirs(path)
    
        
    args = argparse.ArgumentParser()
    args.add_argument("--n_steps", type=int, default=1000, help="Number of steps for the simulation")
    args.add_argument("--k_reps", type=int, default=2, help="Number of repetitions for each parameter set")
    args.add_argument("--job_id", type=int, default=0, help="Job ID for parallel execution")
    args = args.parse_args()

    n_steps = args.n_steps
    k_reps = args.k_reps
    job_id = args.job_id

    headers = ["job_id", "rep", "n_steps", "theta_1", "theta_2", "theta_3", "theta_4", "theta_5", "theta_6", "modularity", "proportion_pure_edges", "mean_majority_proportion"]
    print_headers = True
    mode = "w"

    ETA_PLUS     = np.linspace(0.5, 1.0, 11)
    ETA_MINUS    = np.linspace(0.0, 0.5, 11)
    
    LAMBDA_PLUS  = np.linspace(0.5, 2.0, 4)
    LAMBDA_MINUS = [0.2, 0.5]{}
    # GAMMA_PLUS   = np.linspace(0.0, 0.5, 3)
    # GAMMA_MINUS  = np.linspace(0.0, 0.2, 3)
    GAMMA_PLUS = [0.2]
    GAMMA_MINUS = [0.1]

    THETA = product(ETA_PLUS, ETA_MINUS, LAMBDA_PLUS, LAMBDA_MINUS, GAMMA_PLUS, GAMMA_MINUS)
    
    THETA = list(THETA)
    
    # randomly shuffle THETA inplace so that different jobs are likely to be working on different parameter sets, helps with load balancing.
    np.random.shuffle(THETA)
    
    for rep in tqdm.tqdm(range(k_reps), desc="Repetitions"):
        for eta_plus, eta_minus, lambda_plus, lambda_minus, gamma_plus, gamma_minus in THETA:
            
            if (eta_plus < eta_minus) or (lambda_plus < lambda_minus): 
                continue
            
            if not np.isclose(eta_plus + eta_minus, 1.0) :
                continue

            H = xgi.Hypergraph([[0, 1], [2, 3]])
            H.set_node_attributes({0 : 0, 1 : 0, 2 : 1, 3 : 1}, name = "label")
            growing_hypergraph = GH(H, [0, 1], eta_plus, eta_minus)
            growing_hypergraph.add_hyperedge(n_steps, gamma_plus, gamma_minus, lambda_plus, lambda_minus)

            mod       = clique_projection_modularity(growing_hypergraph)
            prop_pure = proportion_pure_edges(growing_hypergraph)
            mmp       = mean_majority_proportion(growing_hypergraph)
            
            print(f"Job ID: {job_id}, Rep: {rep}, Theta: {(eta_plus, eta_minus, lambda_plus, lambda_minus, gamma_plus, gamma_minus)}, Modularity: {mod:.4f}, Proportion Pure Edges: {prop_pure:.4f}, Mean Majority Proportion: {mmp:.4f}")
            
            theta = [eta_plus, eta_minus, lambda_plus, lambda_minus, gamma_plus, gamma_minus]
            df = pd.DataFrame([[job_id, rep, n_steps, *theta, mod, prop_pure, mmp]], columns=headers)
            df.to_csv(f"throughput/community-structure-sims/{job_id}.csv", mode=mode, header=print_headers, index=False, float_format = "%.4f")
            
            mode = "a"
            print_headers = False        