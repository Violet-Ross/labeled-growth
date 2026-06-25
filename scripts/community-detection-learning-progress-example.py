# if using a .ipynb in a folder, this heading is needed before importing
import sys
import os
import scripts.figure_settings as fs

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, project_root)


# in a .py file, just this import works
from src.poisson_hypergraph import GH # custom hypergraph class
from src.algorithms.simulated_annealing import SimulatedAnnealingApprox
from src.algorithms.gradient_descent import GradientDescent

# generally used packages
import xgi
import csv

from sklearn.metrics import adjusted_rand_score 
import networkx as nx
import numpy as np
from itertools import combinations

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

fs.set_fonts()


def generate_graph_26_starting_nodes(true_theta, timesteps):
    true_p, true_q, gamma_nu, gamma_nr, gamma_eu, gamma_er = true_theta
    H = xgi.Hypergraph([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]])
    H.set_node_attributes({0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:1,14:1,15:1,16:1,17:1,18:1,19:1,20:1,21:1,22:1,23:1,24:1,25:1}, name="label")
    g = GH(H, [0,1], true_p, true_q)
    g.add_hyperedge(timesteps, gamma_nu, gamma_nr, gamma_eu, gamma_er)

    return g

def initial_condition_GH(theta, timesteps):
    eta_plus, eta_minus, gamma_nu, gamma_nr, gamma_eu, gamma_er = theta
    H = xgi.Hypergraph([[0,1,2], [3, 4, 5]])
    H.set_node_attributes({0:0,1:0,2:0,3:0,4:0,5:1}, name="label")
    g = GH(H,[0, 1], eta_plus , eta_minus)
    g.add_hyperedge(timesteps, gamma_nu , gamma_nr , gamma_eu , gamma_er)
    return g



# Modularity Maximization Implementation

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

    partition = nx.community.greedy_modularity_communities(G, best_n=2, weight="weight")
    z = np.array([0 if node in partition[0] else 1 for node in G.nodes()])

    print(len(z))

    return z

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

    from sklearn.cluster import SpectralClustering

    adjacency_matrix = nx.to_numpy_array(G, weight="weight")
    spectral = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=0)
    labels = spectral.fit_predict(adjacency_matrix)

    return labels

if __name__ == "__main__":

    # np.random.seed(12345)
    # define params for hypergraph generation:
    eta_plus = .9 # like-labeled copy param
    eta_minus = .1 # unlike-labeled copy param
    lambda_plus = 1 # like-labeled external param
    lambda_minus = .25 # unlike-labeled external param
    mu_plus = .2 # like-labeled novel param
    mu_minus = .1 # unlike-labeled novel param

    true_theta = [eta_plus, eta_minus, mu_plus, mu_minus, lambda_plus, lambda_minus]

    # number of edges added to synthetic hypergraph
    timesteps = 198

    g = initial_condition_GH(true_theta, timesteps)
        
    
    # in my code, frequently titled "true_theta"
    _eta_plus = .9 # like-labeled copy param
    _eta_minus = .1 # unlike-labeled copy param
    _lambda_plus = 1 # like-labeled external param
    _lambda_minus = .25 # unlike-labeled external param
    _mu_plus = .2 # like-labeled novel param
    _mu_minus = .1 # unlike-labeled novel param

    # combine into format for algorithm
    guessed_theta = [_eta_plus, _eta_minus, _mu_plus, _mu_minus, _lambda_plus, _lambda_minus]

    sa = SimulatedAnnealingApprox(g, guessed_theta, approx=100, novel=True)

    algo_timesteps = len(g.nodes)*20

    for step_num in range(algo_timesteps):
        sa.step()
        if (step_num+1) % (algo_timesteps//10) == 0:
            print(f"Completed {(step_num+1)/algo_timesteps:.0%} steps, current ll/edge: {sa.likelihoods_per_step[-1]/len(g.H.edges):.4f}")
            
        

    m = len(g.H.edges)

    max_ll_index = np.argmax(sa.likelihoods_per_step)

    fig, axarr = plt.subplots(1, 2, figsize=(10, 4))
    axarr[0].plot(np.array(sa.likelihoods_per_step)/m, color = "steelblue")
    axarr[0].set_ylabel("Likelihood per edge")
    axarr[0].set_xlabel("Step")

    axarr[0].scatter(max_ll_index, sa.likelihoods_per_step[max_ll_index]/m, color='black', label='Max LL', zorder = 10)
    axarr[0].legend()

    axarr[1].plot(sa.aris_per_step, color = "steelblue")
    axarr[1].set_ylabel("Adjusted Rand Index (ARI)")
    axarr[1].set_xlabel("Step")
    axarr[1].scatter(max_ll_index, sa.aris_per_step[max_ll_index], color='black', label='Max LL', zorder = 10)
    axarr[1].legend()

    modularity_labels = clique_projection_modularity_maximization_algo(g)
    mod_ari = adjusted_rand_score(g.get_labels(), modularity_labels)
    
    spectral_labels = spectral_clustering_comparison(g)
    spectral_ari = adjusted_rand_score(g.get_labels(), spectral_labels)

    # axarr[1].plot([0, algo_timesteps], [mod_ari, mod_ari], color='black', label='Modularity ARI', linestyle='--', zorder = -10)
    # axarr[1].legend()
    axarr[1].plot([0, algo_timesteps], [spectral_ari, spectral_ari], color='black', label='Spectral Clustering', linestyle='--', zorder = -10)
    axarr[1].legend()
    
    
    plt.tight_layout()

    plt.savefig("fig/community-detection-learning-progress.png", dpi=300, bbox_inches="tight")
    
    tex_dict = {
        "\simannviznumsteps": algo_timesteps,
        "\simannviznumnodes": len(g.nodes),
        "\simannviznumedges": len(g.H.edges),
        "\simannvizmaxll": f"{sa.likelihoods_per_step[max_ll_index]/m:.2f}",
        "\simannvizmaxllari": f"{sa.aris_per_step[max_ll_index]:.2f}",
        "\simannvizmodularityari": f"{mod_ari:.2f}",
        "\simannvizsamecopyrate": f"{eta_plus:.2f}",
        "\simannvizoppcopyrate": f"{eta_minus:.2f}",
        "\simannvizsameextantrate": f"{lambda_plus:.2f}",
        "\simannvizoppextantrate": f"{lambda_minus:.2f}",
        "\simannvizsamenovelrate": f"{mu_plus:.2f}",
        "\simannvizoppnovelrate": f"{mu_minus:.2f}",
        "\simannvizsamecopyrateguess": f"{_eta_plus:.2f}",
        "\simannvizoppcopyrateguess": f"{_eta_minus:.2f}",
        "\simannvizsameextantrateguess": f"{_lambda_plus:.2f}",
        "\simannvizoppextantrateguess": f"{_lambda_minus:.2f}",
        "\simannvizsamenovelrateguess": f"{_mu_plus:.2f}",
        "\simannvizoppnovelrateguess": f"{_mu_minus:.2f}"
    }
    
    with open("paper/community-detection-learning-progress.tex", "w") as f:
        for key, value in tex_dict.items():
            f.write(f"\\newcommand{{{key}}}{{{value}}}\n")
    