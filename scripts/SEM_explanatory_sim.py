from src.poisson_hypergraph import GH
import numpy as np
import xgi
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import csv
import seaborn as sns

def e_prime_prob(growing_hypergraph, e_prime_index, theta):
    edges = growing_hypergraph.get_edges()
    e_prime = edges[e_prime_index]
    summation = 0
    for e_index in range(e_prime_index):
        e = edges[e_index]
        for u_index in e.intersection(e_prime):
            summation += growing_hypergraph.likelihood(e_index, u_index, e_prime_index, theta)
    return summation

def p_q_likelihoods(growing_hypergraph, grain, gammas):
    gamma_nu, gamma_nr, gamma_eu, gamma_er = gammas

    edges = growing_hypergraph.get_edges()
    step = 1 / grain

    P = np.linspace(0 + step, 1 - step, num = grain)
    Q = np.linspace(0 + step, 1 - step, num = grain)
    # P = np.linspace(0, 1, num = grain)
    # Q = np.linspace(0, 1, num = grain)
    
    likelihood_values = np.zeros((grain, grain))

    for p_index, p in enumerate(P):
        for q_index, q in enumerate(Q):
            theta = [p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er]
            this_likelihood = 0
            for e_prime_index in range(1, len(edges)):
                this_likelihood += np.log(e_prime_prob(growing_hypergraph, e_prime_index, theta))
            likelihood_values[q_index, p_index] = this_likelihood
    
    q_max, p_max = np.unravel_index(np.argmax(likelihood_values, axis=None), likelihood_values.shape)
    
    return likelihood_values, p_max, q_max

def gamma_ext_likelihoods(growing_hypergraph, grain, other_thetas):
    p, q, gamma_nu, gamma_nr = other_thetas

    edges = growing_hypergraph.get_edges()
    step = 1 / grain

    # GEU = np.linspace(0 + step, 1 - step, num = grain)
    # GER = np.linspace(0 + step, 1 - step, num = grain)
    GEU = np.linspace(0, 1, num = grain)
    GER = np.linspace(0, 1, num = grain)

    likelihood_values = np.zeros((grain, grain))

    for geu_index, gamma_eu in enumerate(GEU):
        for ger_index, gamma_er in enumerate(GER):
            theta = [p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er]
            this_likelihood = 0
            for e_prime_index in range(1, len(edges)):
                this_likelihood += np.log(e_prime_prob(growing_hypergraph, e_prime_index, theta))
            likelihood_values[ger_index, geu_index] = this_likelihood
    
    ger_max, geu_max = np.unravel_index(np.argmax(likelihood_values, axis=None), likelihood_values.shape)
    
    return likelihood_values, geu_max, ger_max

def gamma_new_likelihoods(growing_hypergraph, grain, other_thetas):
    p, q, gamma_eu, gamma_er = other_thetas

    edges = growing_hypergraph.get_edges()
    step = 1 / grain

    # GNU = np.linspace(0 + step, 1 - step, num = grain)
    # GNR = np.linspace(0 + step, 1 - step, num = grain)
    GNU = np.linspace(0, 1, num = grain)
    GNR = np.linspace(0, 1, num = grain)

    likelihood_values = np.zeros((grain, grain))

    for gnu_index, gamma_nu in enumerate(GNU):
        for gnr_index, gamma_nr in enumerate(GNR):
            theta = [p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er]
            this_likelihood = 0
            for e_prime_index in range(1, len(edges)):
                this_likelihood += np.log(e_prime_prob(growing_hypergraph, e_prime_index, theta))
            likelihood_values[gnr_index, gnu_index] = this_likelihood
    
    gnr_max, gnu_max = np.unravel_index(np.argmax(likelihood_values, axis=None), likelihood_values.shape)
    
    return likelihood_values, gnu_max, gnr_max

def generate_hypergraph_big(theta, size):
    H = xgi.Hypergraph([[0, 1], [0, 2, 3, 4], [1, 5, 6, 7, 8]])
    H.set_node_attributes({0 : 0, 1 : 1, 2 : 0, 3 : 1, 4 : 0, 5 : 1, 6 : 0, 7 : 1, 8 : 0}, name = "label")
    growing_hypergraph = GH(H, [0, 1], theta[0], theta[1])
    growing_hypergraph.add_hyperedge(size, theta[2], theta[3], theta[4], theta[5])
    return growing_hypergraph

def experiment(theta, timesteps, grain):
    p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er = theta
    growing_hypergraph = generate_hypergraph_big(theta, timesteps)

    # get p, q lik values
    lik_values_pq, p_max, q_max = p_q_likelihoods(growing_hypergraph, grain, [gamma_nu, gamma_nr, gamma_eu, gamma_er])

    # get gamma ext lik values
    lik_values_ext, geu_max, ger_max = gamma_ext_likelihoods(growing_hypergraph, grain, [p, q, gamma_nu, gamma_nr])

    # get gamma new lik values
    lik_values_new, gnu_max, gnr_max = gamma_new_likelihoods(growing_hypergraph, grain, [p, q, gamma_eu, gamma_er])


    return lik_values_pq, p_max, q_max, lik_values_ext, geu_max, ger_max, lik_values_new, gnu_max, gnr_max


lik_values_pq, p_max, q_max, lik_values_ext, geu_max, ger_max, lik_values_new, gnu_max, gnr_max = experiment([0.6, 0.4, 0.75, 0.25, 0.5, 0.5], 500, 50)
#lik_values_pq, p_max, q_max, lik_values_ext, geu_max, ger_max, lik_values_new, gnu_max, gnr_max = experiment([0.6, 0.4, 0.75, 0.25, 0.5, 0.5], 20, 5)


with open('throughput/SEM_explanatory_pq.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(lik_values_pq)

with open('throughput/SEM_explanatory_ext.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(lik_values_ext)

with open('throughput/SEM_explanatory_new.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(lik_values_new)