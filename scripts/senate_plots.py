from src.poisson_hypergraph import GH
from src.algorithms.sem import sem_functions
import xgi
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from collections import Counter
from itertools import combinations

H = xgi.read_json("throughput/senate_bills.json", nodetype=int)
growing_hypergraph = GH(H, [0, 1], 0, 0)

def SEM_viz(estimates1):
    sns.set_style("whitegrid")
    sns.set_palette("Dark2")
    
    fig, axs = plt.subplots(1, 3, sharex = True, sharey = True)
    fig.set_figwidth(15)
    fig.set_figheight(2.5)
    #plt.setp(axs, ylim=(0, 1))
    plt.subplots_adjust(hspace = 0.4)

    plt.rcParams.update({'font.size': 14})
    plt.rc('xtick', labelsize=10) 
    plt.rc('ytick', labelsize=10) 
    
    axs[0].set_title('$\hat{p} = $' f'{estimates1[-1][1]:.2f}' ", " '$\hat{q} = $' f'{estimates1[-1][2]:.2f}')
    axs[0].plot(estimates1[:, 0], estimates1[:, 1], label = "$\hat{p}$")
    axs[0].plot(estimates1[:, 0], estimates1[:, 2], label = "$\hat{q}$")
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol = 2)

    axs[1].set_title('$\hat{\gamma}_{e, z_u} = $' f'{estimates1[-1][5]:.2f}' ", " '$\hat{\gamma}_{e, \\bar{z}_u} = $' f'{estimates1[-1][6]:.2f}')
    axs[1].plot(estimates1[:, 0], estimates1[:, 5], label = "$\hat{\gamma}_{e, z_u}$")
    axs[1].plot(estimates1[:, 0], estimates1[:, 6], label = "$\hat{\gamma}_{e, \\bar{z}_u}$")
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol = 2)
    
    axs[2].set_title('$\hat{\gamma}_{n, z_u} = $' f'{estimates1[-1][3]:.2f}' ", " '$\hat{\gamma}_{n, \\bar{z}_u} = $' f'{estimates1[-1][4]:.2f}')
    axs[2].plot(estimates1[:, 0], estimates1[:, 3], label = "$\hat{\gamma}_{n, z_u}$")
    axs[2].plot(estimates1[:, 0], estimates1[:, 4], label = "$\hat{\gamma}_{n, \\bar{z}_u}$")
    axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol = 2)

    axs[0].set_xlabel("iterations")
    axs[1].set_xlabel("iterations")
    axs[2].set_xlabel("iterations")

    fig.savefig('fig/senate_SEM.png', dpi=300, bbox_inches="tight")

## Degree distribution of senate data
centers, heights = xgi.degree_histogram(H)
list_of_edges_sizes = H.edges.size.aslist()
size_counts = dict(Counter(list_of_edges_sizes))

sns.set_style("whitegrid")
sns.set_palette("Dark2")

plt.rcParams.update({'font.size': 20})
plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15) 

fig, axs = plt.subplots(1, 2, sharey=True)
fig.set_figwidth(20)
fig.set_figheight(10)
axs[0].scatter(centers, np.array(heights) / len(H.nodes))
#axs[0].plot(range(1, 10**3), [x ** (-2) for x in range(1, 10**3)], c = "black")
axs[0].set_ylabel("P(k)")
axs[0].set_xlabel("k")
axs[0].set_xscale('log')
axs[0].set_yscale('log')

axs[1].scatter(list(size_counts.keys()), np.array(list(size_counts.values())) / len(H.nodes))
axs[1].set_ylabel("P(s)")
axs[1].set_xlabel("s")
axs[1].set_yscale('log')

plt.savefig('fig/senate_degree_and_edge_distribution.png', dpi=300, bbox_inches="tight")

## SEM on senate data
sem = sem_functions()
s_intial = np.array([1, 2, 1, 2, 0.5, 0.5, 0.5, 0.5])
estimates = sem.SEM_without_likelihood(growing_hypergraph, s_intial, 3000, 0.01, 0.001)

SEM_viz(np.array(estimates))