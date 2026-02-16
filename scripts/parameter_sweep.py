# Evaluate algorithm performance on different parameters



import sys
import xgi
from poisson_hypergraph import GH
from NMI_func import NMI
from gradient_descent import GradientDescent
import numpy as np
import csv
import random
import networkx as nx
from simulated_annealing import simulated_annealing_full_likelihood, simulated_annealing_approx_likelihood, simulated_annealing_likelihood_batch_approx, SimulatedAnnealingApprox
import time
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
# takes in the algorithm and the timesteps to generate the graph from
# if best likelihood true, returns the best label set from entire run... expensive
# lambda and mu have default values...
# all algorithm results are averaged

parameter_index = int(sys.argv[1])
GRID_SIZE = 10
NUM_RUNS_PER_PARAM_SET = 50

def generate_graph_26_starting_nodes(true_theta, timesteps):
    true_p, true_q, gamma_nu, gamma_nr, gamma_eu, gamma_er = true_theta
    H = xgi.Hypergraph([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]])
    H.set_node_attributes({0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:1,14:1,15:1,16:1,17:1,18:1,19:1,20:1,21:1,22:1,23:1,24:1,25:1}, name="label")
    g=GH(H, [0,1], true_p, true_q)
    g.add_hyperedge(timesteps, gamma_nu, gamma_nr, gamma_eu, gamma_er)

    return g

# create param list
def parameter_list_init():
    param_list = []
    for a in range((GRID_SIZE)):
        eta_plus = .5 + a/(GRID_SIZE)/2
        eta_plus = round(eta_plus, 2)
        eta_minus = .5 - a/(GRID_SIZE)/2
        eta_minus = round(eta_minus, 2)
        for b in range((GRID_SIZE)):
            lambda_plus = 1 + b/(GRID_SIZE+1)
            lambda_plus = round(lambda_plus, 1)
            lambda_minus = 1 - b/(GRID_SIZE+1)
            lambda_minus = round(lambda_minus, 1)

            param_list.append([eta_plus, eta_minus, lambda_plus, lambda_minus])

    return param_list

params = parameter_list_init()

def parameter_sweep(algorithm, timesteps, best_likelihood):
    true_theta = [params[parameter_index][0], params[parameter_index][1], .001, .001, params[parameter_index][2], params[parameter_index][3]]
    results_ari = []
    results_LL = []

    for _ in range(NUM_RUNS_PER_PARAM_SET):
        # true_theta_generate = [.9, .1, .001, .001, 1, .25] # for wrong params only
        g = generate_graph_26_starting_nodes(true_theta, timesteps)

        labels = algorithm(g, true_theta)
        results_ari.append(adjusted_rand_score(labels, g.get_labels()))
        results_LL.append(g.total_log_likelihood(true_theta, labels))

    mean_ari = np.mean(results_ari)
    mean_LL = np.mean(results_LL)

    return mean_ari, mean_LL

def parameter_sweep_gradient_descent(timesteps):
    true_theta = [params[parameter_index][0], params[parameter_index][1], .001, .001, params[parameter_index][2], params[parameter_index][3]]
    results_ari = []
    results_LL = []

    for _ in range(NUM_RUNS_PER_PARAM_SET):
        # true_theta_generate = [.9, .1, .001, .001, 1, .25] # for wrong params only
        g = generate_graph_26_starting_nodes(true_theta, timesteps)

        gd = GradientDescent(true_theta, g, .001, (.9,.99))

        gd.run(200)

        # labels = algorithm(g, true_theta, best_likelihood)
        results_ari.append(gd.label_aris[-1])
        # results_LL.append(gd.label_LLs_converted[-1])

    mean_ari = np.mean(results_ari)

    return mean_ari


def parameter_sweep_greedy(timesteps):
    true_theta = [params[parameter_index][0], params[parameter_index][1], .3, .1, params[parameter_index][2], params[parameter_index][3]]
    results_ari = []
    results_LL = []

    for _ in range(NUM_RUNS_PER_PARAM_SET):
        # true_theta_generate = [.9, .1, .001, .001, 1, .25] # for wrong params only
        g = generate_graph_26_starting_nodes(true_theta, timesteps)

        sa = SimulatedAnnealingApprox(g, true_theta)

        for _ in range(len(g.nodes)*50):
            sa.step()

        # labels = algorithm(g, true_theta, best_likelihood)
        results_ari.append(sa.aris_per_step)
        results_LL.append(sa.likelihoods_per_step)

        print(sa.aris_per_step[-1])

    return results_ari, results_LL


def parameter_sweep_modularity_maximization(timesteps):
    true_theta = [params[parameter_index][0], params[parameter_index][1], .5, .2, params[parameter_index][2], params[parameter_index][3]]
    results_ari = []

    for _ in range(NUM_RUNS_PER_PARAM_SET):
        # true_theta_generate = [.9, .1, .001, .001, 1, .25] # for wrong params only
        g = generate_graph_26_starting_nodes(true_theta, timesteps)

        labels = clique_projection_modularity_maximization_algo(g)
        results_ari.append(adjusted_rand_score(g.get_labels(), labels))

    mean_ari = np.mean(results_ari)

    return mean_ari

print(params[parameter_index])

# used as a baseline to justify the algorithm
from itertools import combinations
def clique_projection_modularity_maximization_algo(g):
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

    partition = nx.community.greedy_modularity_communities(G, best_n=2)
    z = np.array([0 if node in partition[0] else 1 for node in G.nodes()])

    return z


import time
def parameter_sweep_thesis(timesteps):
    true_theta = [params[parameter_index][0], params[parameter_index][1], .5, .2, params[parameter_index][2], params[parameter_index][3]]
    results_simulated_annealing_ari = []
    results_simulated_annealing_max_ari = []
    results_gradient_descent_ari = []
    results_modularity_ari = []

    modularity_time = []
    simulated_annealing_time = []
    gradient_descent_time = []

    for _ in range(NUM_RUNS_PER_PARAM_SET):
        
        g = generate_graph_26_starting_nodes(true_theta, timesteps)
        algo_theta = true_theta
        # algo_theta = [.8, .2, .3, .1, 1.6, .4] # for wrong params mode only

        start_time = time.time()
        labels = clique_projection_modularity_maximization_algo(g)
        results_modularity_ari.append(adjusted_rand_score(g.get_labels(), labels))
        modularity_time.append(time.time() - start_time)

        start_time = time.time()


        sa = SimulatedAnnealingApprox(g, algo_theta)

        for _ in range(len(g.nodes)*20):
            sa.step()

        simulated_annealing_time.append(time.time()-start_time)

        results_simulated_annealing_ari.append(sa.aris_per_step[-1])
        results_simulated_annealing_max_ari.append(sa.max_LL_corresponding_ari)
        
        start_time = time.time()
        gd = GradientDescent(algo_theta, g, .001, (.9,.99))

        gd.run(1000)
        gradient_descent_time.append(time.time()-start_time)

        # labels = algorithm(g, true_theta, best_likelihood)
        results_gradient_descent_ari.append(gd.label_aris[-1])

    
    from statistics import mean
    with open('thesis_parameter_sweep_25_steps.csv', 'a', newline="") as file:
        writer = csv.writer(file)
        row = [mean(modularity_time), mean(simulated_annealing_time), mean(gradient_descent_time)]
        writer.writerows([row])

    return [np.mean(results_simulated_annealing_ari), np.mean(results_simulated_annealing_max_ari), np.mean(results_gradient_descent_ari), np.mean(results_modularity_ari)]


# nmi, LL = parameter_sweep(greedy_posterior_prob_with_e_threshold, 30, True)
time.sleep(-1*parameter_index)
data = parameter_sweep_thesis(50)


with open('thesis_parameter_sweep_25_steps.csv', 'a', newline="") as file:
    writer = csv.writer(file)
    row = [parameter_index]
    row = row + data
    row = row + params[parameter_index]
    writer.writerows([row])



# ari, LL = parameter_sweep(simulated_annealing_approx_likelihood, 20, True)
# ari_list, LL_list = parameter_sweep_greedy(50)
# # ari, LL = parameter_sweep_gradient_descent(50)
# # ari = parameter_sweep_gradient_descent(50)

# # ari = parameter_sweep_modularity_maximization(50)

# aris = []
# for lis in ari_list:
#     aris.append(lis[-1])

# LLs = []
# for lis in LL_list:
#     LLs.append(lis[-1])

# ari = np.mean(aris)
# LL = np.mean(LLs)

# # # write results to CSV
# with open('simulated_annealing.csv', 'a', newline="") as file:
#     writer = csv.writer(file)
#     row = [parameter_index, ari, LL]
#     row = row + params[parameter_index]
#     writer.writerows([row])


# if (parameter_index == -25):

#     with open("simulated_annealing_full_runs_LL.csv", 'a', newline="") as file:
#         writer = csv.writer(file)
#         writer.writerows([[parameter_index] + params[parameter_index]])
#         writer.writerows(LL_list)

#     with open("simulated_annealing_full_runs_ari.csv", 'a', newline="") as file:
#         writer = csv.writer(file)
#         writer.writerows([[parameter_index] + params[parameter_index]])
#         writer.writerows(ari_list)

# with open("modularity_maximization_ari.csv", 'a', newline="") as file:
#     writer = csv.writer(file)
#     row = [parameter_index, ari, ari]
#     row = row+params[parameter_index]
#     writer.writerows([row])