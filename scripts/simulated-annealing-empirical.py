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


data_sets = {
    "senate_bills": "senate_bills",
    "highschool_class": "highschool",
    "highschool_gender": "highschool_gender",
    "primaryschool_gender": "primaryschool_gender",
    "primaryschool_class": "primaryschool",
    "gender_coauth": "gender_coauth_shrunk"
}

if __name__ == "__main__":
    
    # figure out which data set we are using
    args = argparse.ArgumentParser(description='Run simulated annealing on a user-specified prepared data set.')
    args.add_argument('--data_set', type=str, choices=data_sets.keys(), help='The data set to run simulated annealing on. Must be one of: ' + ', '.join(data_sets.keys()))
    args.add_argument("--job_id", type=int, help="The job ID for this run, used to differentiate output files when running multiple jobs.")
    args.add_argument("--results_folder", type=str, help="Folder to store results (defaults to data set name)", default=None)
    args.add_argument("--params", type=float, nargs=6, help="Specified simulated annealingparameters to override default.  6-tuple of floats in the form (p+, p-, a+, a-, r+, r-)", default=None)

    args = args.parse_args()
    
    data_set = args.data_set
    job_id = args.job_id
    params = args.params

    results_folder = data_sets[data_set] # set default name of results folder to the title of dataset
    if args.results_folder != None:
        results_folder = args.results_folder # override if provided

    # order is copy, novel, extant
    true_theta = [.95, .05, .001, .001, 1, .01] # default params
    if params != None: # override default params if user-specified
        true_theta = params
    
    path = f"throughput/simulated_annealing/{results_folder}"
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + "/labels", exist_ok=True)
    os.makedirs(path + "/metrics", exist_ok=True)
    
    # initialize 
    # H = pkl.load(open(f"throughput/{data_sets[data_set]}.pkl", "rb")) unsure of why this is used
    H = xgi.read_json(f"throughput/{data_sets[data_set]}.json", nodetype=int)
    g = GH(H, [0, 1], 0, 0)
    
    
    
    print("initializing simulated annealing...")
    sa = SimulatedAnnealingApprox(g, true_theta)

    best_likelihood = -float('inf')
    best_ari = -float('inf')
    best_labels = None
    
    mode = "w"
    header = True
    print("running simulated annealing...")
    
    num_steps = len(g.nodes)*20
    
    for step_num in tqdm(range(num_steps)):
        print("step num: " + str(step_num))
        sa.step()
        
        likelihood = sa.likelihoods_per_step[-1]
        if likelihood > best_likelihood:
            best_likelihood = likelihood
            best_ari = sa.aris_per_step[-1]
            best_labels = sa.labels
            
            df = pd.DataFrame({
                "labels"   : [best_labels],
                "step_num" : [step_num]
            })
                        
            df.to_csv(f'{path}/labels/{job_id}.csv', index=False, mode = mode, header = header)
            
            df = pd.DataFrame({
                "likelihood" : [best_likelihood],
                "ari"        : [best_ari] ,
                "step_num"   : [step_num]
            })
            
            df.to_csv(f'{path}/metrics/{job_id}.csv', index=False, mode = mode, header = header)
            
            mode = "a"
            header = False  