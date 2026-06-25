from cmath import rect
import sys 
sys.path.append("src")
sys.path.append("scripts")


import numpy as np 
from itertools import product
from matplotlib import pyplot as plt
from scipy.special import binom
from toy_edge_size_simulator import ToySimulator
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import factorial
from linear_map import matrix_of_linear_map
import matplotlib.patches as patches

import figure_settings as fs

plt.style.use('seaborn-v0_8-whitegrid')

fs.set_fonts()


def get_dist(TS, window, k_max):
    C = np.zeros((k_max + 1, k_max + 1))
    for e in TS.edge_list[-window:]: 
        if e[0] <= k_max and e[1] <= k_max:
            C[e[0], e[1]] += 1
    return C / C.sum() 



if __name__ == "__main__":
    
    # np.random.seed(123)
    
    # spectral gap analysis
    
    k_max = 12
    # k_max = 8
    
    grid_resolution = 9
    gaps = np.zeros((grid_resolution, grid_resolution))
    
    ETA_PLUS = np.linspace(0.1, 0.9, grid_resolution)
    ETA_MINUS = np.linspace(0.1, 0.9, grid_resolution)
    
    for i, eta_plus in enumerate(ETA_PLUS):
        for j, eta_minus in enumerate(ETA_MINUS):
            print(f"Computing spectral gap for eta_plus = {eta_plus:.1f}, eta_minus = {eta_minus:.1f}")
            theta = [eta_plus, eta_minus, 0.5, 0.2, 0.4, 0.2]
            M = matrix_of_linear_map(k_max, theta)
            vals, E = np.linalg.eig(M)
            idx = vals.argsort()[::-1]
            spectral_gap = np.abs(vals[idx[0]]) - np.abs(vals[idx[1]])
            gaps[j, i] = spectral_gap
            
    fig, ax = plt.subplots(1, 1, figsize = (6, 6))
    im = ax.imshow(gaps, cmap = "viridis", interpolation = "nearest", origin = "lower")
    ax.set_title("Spectral gap of linear map")
    ax.set_xlabel(r"$\eta_+$")
    ax.set_ylabel(r"$\eta_-$")
    ax.set_xticks(range(grid_resolution))
    ax.set_xticklabels([f"{x:.1f}" for x in np.linspace(0.1, 0.9, grid_resolution)])
    ax.set_yticks(range(grid_resolution))
    ax.set_yticklabels([f"{x:.1f}" for x in np.linspace(0.1, 0.9, grid_resolution)])
    plt.colorbar(im, ax = ax)
    plt.savefig("fig/spectral_gap.png", dpi = 300)
            
    
    
    
    
    # 
    
    # eta_plus_index = 8
    # eta_minus_index = 1
    # eta_plus_index = 8
    eta_plus_index = 7
    eta_minus_index = 0
    eta_plus = ETA_PLUS[eta_plus_index]
    eta_minus = ETA_MINUS[eta_minus_index]
    
    
    
    
    
    theta = [eta_plus, eta_minus, 0.5, 0.2, 0.4, 0.2]
    
    # FIRST: spectral analysis of the linear operator governing the system dynamics
    
    M = matrix_of_linear_map(k_max, theta)
    
    # column sums should be equal to 1
    
    fig, ax = plt.subplots(1, 1, figsize = (6, 6))
    
    ax.imshow(M, cmap = "viridis", interpolation = "nearest", origin = "lower")
    ax.set_title("Matrix of Linear Map (shouldn't this be symmetric?)")
    ax.set_xlabel(r"$k_0', k_1'$ (new edge size)")
    ax.set_ylabel(r"$k_0, k_1$ (old edge size)")
    plt.savefig("fig/linear_map_matrix.png", dpi = 300)
    
    # time for eigenstuff
    vals, E = np.linalg.eig(M)    

    fig, ax = plt.subplots(1, 1, figsize = (6, 6))
    ax.plot(vals.real, vals.imag, "o")
    ax.set_title("Eigenvalues of Linear Map")
    ax.set_xlabel("Real part")
    ax.set_ylabel("Imaginary part")
    plt.savefig("fig/linear_map_eigenvalues.png", dpi = 300)

    # retrieve the top eigenvectors 
    idx = vals.argsort()[::-1]
    vals = vals[idx]
    E = E[:, idx]
    
    eigvecs = []
    for i in range(2):
        vi = E[:, i].real
        vi = vi / vi.sum()
        vi = vi / np.abs(vi).sum()
        eigvecs.append(vi.reshape((k_max + 1, k_max + 1)))
    
    # SECOND: simulation with the same parameters
    
    n_steps = int(1e7)
    num_projections = 20
    logspace = np.logspace(0, np.log10(n_steps+1), num_projections, dtype = int)

    # project_every = 1000
    window = 1000
    
    TS = ToySimulator(edge_list = [[5,5]], theta = theta, force_label = None)

    projections = {i: [] for i in range(2)}
    projection_timesteps = []
    
    # print(Cz)
    # print((eigvecs[0]**2).sum())
    
    for j in range(1, n_steps+1): 
        TS.simulate(n_samples = 1)
            
        if j in logspace:
            C = get_dist(TS, window, k_max)
            
            # experimental results here are a bit weird since the 
            # projection onto the third eigenvector doesn't seem to decay. 
            # Are the eigenvectors not orthogonal? 
            for i in range(2): 
                
                C_ = C / np.sqrt((C**2).sum())
                E_ = eigvecs[i]
                E_norm = E_ / np.sqrt((E_**2).sum())
                         
                projections[i].append((E_norm*C_).sum())
            projection_timesteps.append(j)
            
    
    # fig = plt.figure(figsize = (11, 7))    
    fig = plt.figure(figsize = (13, 7))
    widths = [1,  1, 1]
    gs = GridSpec(2, 3, figure=fig, width_ratios=widths, hspace = 0.35, wspace = 0.2)
    
    axes = []
    
    for i in range(3):
        axes.append(fig.add_subplot(gs[0, i]))
    
    axes.append(fig.add_subplot(gs[1, 0:2]))
    axes.append(fig.add_subplot(gs[1, 2]))
    
    
    
    # num_eigvecs = 2
    
    v_min = -0.05
    vmax  = 0.05
    
    # fig, ax = plt.subplots(1, 4, figsize = (18, 6))
    
    colors = ["#055775", "darkgrey", "lightgrey"]
    letter_labels = ["(b)", "(c)"]
    
    for i in range(2):
        vi = np.real(E[:, i])
        val = np.real(vals[i]) if i > 0 else 1.0
        vi = vi / vi.sum()
        vi = vi / np.abs(vi).sum()
        P = vi.reshape((k_max + 1, k_max + 1))
        
        ax = axes[i+1]
        im = ax.imshow(P, cmap = "BrBG", interpolation = "none", vmin = v_min, vmax = vmax, origin = "lower")
        ax.grid(False)
        ax.set_title(rf"{letter_labels[i]}: $\mathbf{{v}}_{{{i+1}}}$ ($\lambda_{{{i+1}}} = {val:.2f}$)")
        ax.set_xlabel(r"$k_0$")
        if i == 0: 
            ax.set_ylabel(r"$k_1$")
            
        ax.set_xticks(range(0, k_max + 1, 3))
        ax.set_yticks(range(0, k_max + 1, 3))
        
        ax = axes[3]
        ax.plot(projection_timesteps, projections[i], label = fr"$\mathbf{{v}}_{{{i+1}}}$", color = colors[i])
        ax.legend()
        ax.semilogx()
        ax.set_title("(d): Projection of simulation state onto eigenvectors")
    

    
    
    ax = axes[3]
    # ax.plot([min(projection_timesteps), n_steps], [0, 0], "k--", alpha = 0.5)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cosine similarity")
    # axes[3].set_xlim(window, None)
    
    
    #     fig.colorbar(im, ax = ax[i])
    
    ax = axes[4]
    final_window = n_steps
    C = get_dist(TS, final_window, k_max)
    final = ax.imshow(C / C.sum(), cmap = "Greys", interpolation = "none", vmin = 0, origin = "lower")
    ax.set_title(f"(e): Final simulation state")
    
    ax.set_xlabel(r"$k_0$")
    ax.set_ylabel(r"$k_1$")
    
    ax.set_xticks(range(0, k_max + 1, 3))
    ax.set_yticks(range(0, k_max + 1, 3))
    ax.grid(False)
    
    fig.colorbar(im)
    fig.colorbar(final)
    
    
    ax = axes[0]
    im = ax.imshow(gaps, cmap = "inferno", interpolation = "none", origin = "lower")
    ax.set_title("(a): Spectral gap")
    ax.set_xlabel(r"$\eta_+$")
    ax.set_ylabel(r"$\eta_-$")
    ax.set_xticks(range(grid_resolution))
    ax.set_xticklabels([f"{x:.1f}" for x in np.linspace(0.1, 0.9, grid_resolution)])
    ax.set_yticks(range(grid_resolution))
    ax.set_yticklabels([f"{x:.1f}" for x in np.linspace(0.1, 0.9, grid_resolution)])
    ax.grid(False)
    
    fig.colorbar(im, ax = ax, format = "%.2f")
    
    

    # highlight the eta_plus and eta_minus values used in the simulation
    rect = patches.Rectangle((eta_plus_index - 0.5, eta_minus_index - 0.5), 1, 1, linewidth=2, edgecolor='white', facecolor='none')
    axes[0].add_patch(rect)
    

    plt.savefig("fig/linear_map_eigenvectors.png", dpi = 300, bbox_inches = "tight")

