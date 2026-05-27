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

n_steps = int(25)

params = {
    "theta_1"  : [0.95, 0.05, 0.5, 0.25, 0.5, 0.25], 
    "theta_2"  : [0.3, 0.1, 0.8, 0.4, 0.5, 0.25], 
    # "theta_3"  : [0.5, 0.5, 0.2, 0.1, 0.2, 0.1], 
}

# params = {
#     "theta_1"  : [0.7, 0.3, 0.5, 0.5, 0.2, 0.1], 
#     "theta_2"  : [0.7, 0.3, 0.7, 0.3, 0.2, 0.1], 
#     "theta_3"  : [0.7, 0.3, .9, 0.1, 0.2, 0.1]
# }

fig, ax = plt.subplots(1, len(params), figsize=(12, 5))

for i, theta in enumerate(params.keys()):
    
    eta_plus, eta_minus, lambda_plus, lambda_minus, gamma_plus, gamma_minus = params[theta]
    
    H = xgi.Hypergraph([[0, 1, 2, 3], [4, 5, 6]])
    H.set_node_attributes({0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 1, 5 : 1, 6 : 1}, name = "label")
    growing_hypergraph = GH(H, [0, 1], eta_plus, eta_minus)
    growing_hypergraph.add_hyperedge(n_steps, gamma_plus, gamma_minus, lambda_plus, lambda_minus)
    
    labels = growing_hypergraph.get_labels()
    H = growing_hypergraph.H
    
    
    H.cleanup(isolates = True)
    H.set_edge_attributes({e : H.edges.size(e) for e in H.edges}, name = "size")
    
    @xgi.edgestat_func
    def purity(net, bunch):
        """The purity of a bunch of edges in net."""
        return {e: 1 if len(set(labels[j] for j in net.edges.members()[e])) == 1 else 0 for e in bunch}
    
    
    
    xgi.draw(growing_hypergraph.H, 
             node_fc=labels, 
            #  node_size = H.nodes.degree,
             ax=ax[i],
             node_fc_cmap=plt.cm.gray_r, 
            edge_fc=H.edges.purity,
            dyad_color = H.edges.purity,
            dyad_color_cmap=plt.cm.PiYG,
            edge_fc_cmap=plt.cm.PiYG,
            edge_lw = 0,
            node_ec = "grey", 
            hull = True, 
            alpha = 0.1,
             )
    
    
    title = fr"""
    $\eta_{{+}}$ = {eta_plus:.2f}, $\eta_{{-}}$ = {eta_minus:.2f}
    $\lambda_{{+}}$ = {lambda_plus:.2f}, $\lambda_{{-}}$ = {lambda_minus:.2f}
    """
    
    
    
    ax[i].set_title(title)


plt.savefig("fig/hypergraph_viz.png", dpi=300, bbox_inches="tight")


# UNLABELED VERSION

n_steps = int(10)

theta_1 = [0.9, 0.9, 0.5, 0.5, 0.5, 0.5]
eta_plus, eta_minus, lambda_plus, lambda_minus, gamma_plus, gamma_minus = params[theta]

H = xgi.Hypergraph([[0, 1, 2, 3]])
H.set_node_attributes({0 : 0, 1 : 0, 2 : 0, 3 : 0}, name = "label")
growing_hypergraph = GH(H, [0, 1], eta_plus, eta_minus)
growing_hypergraph.add_hyperedge(n_steps, gamma_plus, gamma_minus, lambda_plus, lambda_minus)
H = growing_hypergraph.H.copy()
H = H.cleanup(isolates = True)


fig, ax = plt.subplots(1, 1, figsize=(6, 5))
xgi.draw(H,
         node_fc="white",
        #  node_size = 100,
         ax=ax,
         node_fc_cmap=plt.cm.gray_r, 
         hull = True,
         edge_fc_cmap = plt.cm.plasma)
plt.savefig("fig/hypergraph_viz_unlabeled.png", dpi=300, bbox_inches="tight")