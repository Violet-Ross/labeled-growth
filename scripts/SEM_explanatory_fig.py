from src.poisson_hypergraph import GH
import numpy as np
import xgi
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import csv
import seaborn as sns

def make_plot(grain, lik_values_pq, p_max, q_max, lik_values_ext, geu_max, ger_max, lik_values_new, gnu_max, gnr_max, true_theta):
    step = 1/grain

    bins_1 = lik_values_pq.shape[0]
    bins_2 = lik_values_ext.shape[0]
    bins_3 = lik_values_new.shape[0]

    P = np.linspace(0, 1, num = bins_1)
    Q = np.linspace(0, 1, num = bins_1)

    GEU = np.linspace(0, 1, num = bins_2)
    GER = np.linspace(0, 1, num = bins_2)

    GNU = np.linspace(0, 1, num = bins_3)
    GNR = np.linspace(0, 1, num = bins_3)

    vmin = min(np.min(lik_values_pq), np.min(lik_values_ext), np.min(lik_values_new))
    vmax = max(np.max(lik_values_pq), np.max(lik_values_ext), np.max(lik_values_new))

    sns.set_style("whitegrid")
    sns.set_palette("Dark2")

    plt.rcParams.update({'font.size': 17})
    plt.rc('xtick', labelsize=13) 
    plt.rc('ytick', labelsize=13) 

    fig4, ax4 = plt.subplots(1, 3, sharex = True, sharey = True)
    fig4.set_figwidth(20)
    fig4.set_figheight(9)
    cmap = cm['viridis']
    cmap.set_bad('black',1.)

    plt.xlim(0, 1)

    im1 = ax4[0].imshow(lik_values_pq, cmap=cmap, interpolation='nearest', origin = "lower", vmin = vmin, vmax = vmax)
    ax4[0].set_xticks(ticks = np.linspace(0, 1, 11)*bins_1, labels = np.round(np.linspace(0, 1, 11), 2))
    ax4[0].set_yticks(ticks = np.linspace(0, 1, 11)*bins_1, labels = np.round(np.linspace(0, 1, 11), 2))
    ax4[0].set_xlabel("p")
    ax4[0].set_ylabel("q")
    ax4[0].scatter([true_theta[0] * grain], [true_theta[1] * grain], zorder = 100, color = 'C0', edgecolors="black")
    ax4[0].scatter([p_max], [q_max], zorder = 100, color = 'C1', edgecolors="black")
    ax4[0].contour(np.linspace(0, lik_values_pq.shape[0], num = bins_1), np.linspace(0, lik_values_pq.shape[0], num = bins_1), lik_values_pq, 
                   levels = 20, colors = "white", linestyles = "solid")
    ax4[0].set_title(r"$\hat{p} = $" + f"{P[p_max]:.2f}" + "  |  " r"$\hat{q} = $" + f"{Q[q_max]:.2f}")


    im2 = ax4[1].imshow(lik_values_ext, cmap=cmap, interpolation='nearest', origin = "lower", vmin = vmin, vmax = vmax)
    ax4[1].set_xlabel("$\\hat{\\gamma}_{e, z_u}$")
    ax4[1].set_ylabel("$\\hat{\\gamma}_{e, \\bar{z}_u}$")
    ax4[1].scatter([true_theta[4] * grain], [true_theta[5] * grain], zorder = 100, color = 'C0', edgecolors="black")
    ax4[1].scatter([geu_max], [ger_max], zorder = 100, color = 'C1', edgecolors="black")
    ax4[1].contour(np.linspace(0, lik_values_ext.shape[0], num = bins_1), np.linspace(0, lik_values_ext.shape[0], num = bins_1), lik_values_ext, 
                   levels = 20, colors = "white", linestyles = "solid")
    ax4[1].set_title(r"$\hat{\gamma}_{e, z_u} = $" + f"{GEU[geu_max]:.2f}" + "  |  " r"$\hat{\gamma}_{e, \bar{z}_u} = $" + f"{GER[ger_max]:.2f}")

    im3 = ax4[2].imshow(lik_values_new, cmap=cmap, interpolation='nearest', origin = "lower", vmin = vmin, vmax = vmax)
    ax4[2].set_xlabel("$\\hat{\\gamma}_{n, z_u}$")
    ax4[2].set_ylabel("$\\hat{\\gamma}_{n, \\bar{z}_u}$")
    ax4[2].scatter([true_theta[2] * grain], [true_theta[3] * grain], zorder = 100, color = 'C0', edgecolors="black")
    ax4[2].scatter([gnu_max], [gnr_max], zorder = 100, color = 'C1', edgecolors="black")
    ax4[2].contour(np.linspace(0, lik_values_new.shape[0], num = bins_1), np.linspace(0, lik_values_new.shape[0], num = bins_1), lik_values_new, 
                   levels = 20, colors = "white", linestyles = "solid")
    ax4[2].set_title(r"$\hat{\gamma}_{n, z_u} = $" + f"{GNU[gnu_max]:.2f}" + "  |  " r"$\hat{\gamma}_{n, \bar{z}_u} = $" + f"{GNR[gnr_max]:.2f}")

    cbar_ax = fig4.add_axes([0.22, 0.10, 0.56, 0.05])
    fig4.colorbar(im1, cax=cbar_ax, orientation='horizontal', label = "Log marginal likelihood")


    fig4.savefig('fig/sem_explanatory.png', dpi=300,  bbox_inches="tight")

lik_values_pq = []
with open('throughput/SEM_explanatory_pq.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        lik_values_pq.append(row)
lik_values_pq = np.array(lik_values_pq).astype(float)

lik_values_ext = []
with open('throughput/SEM_explanatory_ext.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        lik_values_ext.append(row)
lik_values_ext = np.array(lik_values_ext).astype(float)

lik_values_new = []
with open('throughput/SEM_explanatory_new.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        lik_values_new.append(row)
lik_values_new = np.array(lik_values_new).astype(float)

q_max, p_max = np.unravel_index(np.argmax(lik_values_pq, axis=None), lik_values_pq.shape)
ger_max, geu_max = np.unravel_index(np.argmax(lik_values_ext, axis=None), lik_values_ext.shape)
gnr_max, gnu_max = np.unravel_index(np.argmax(lik_values_new, axis=None), lik_values_new.shape)

make_plot(50, lik_values_pq, p_max, q_max, lik_values_ext, geu_max, ger_max, lik_values_new, gnu_max, gnr_max, [0.6, 0.4, 0.75, 0.25, 0.5, 0.5])