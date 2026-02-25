import xgi
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import itertools
import csv

## Plotting with likelihood

def viz_with_likelihood(true_values1, estimates1, liks1, true_values2, estimates2, liks2, true_values3, estimates3, liks3, true_values4, estimates4, liks4):
    sns.set_style("whitegrid")
    sns.set_palette("Dark2")
    
    fig, axs = plt.subplots(4, 4, sharex = True, sharey= "row")
    fig.set_figwidth(20)
    fig.set_figheight(10)
    #fig.tight_layout()
    plt.subplots_adjust(hspace = 0.4)
    plt.setp(axs, ylim=(0, 1.1))

    plt.rcParams.update({'font.size': 17})
    plt.rc('xtick', labelsize=13) 
    plt.rc('ytick', labelsize=13) 

    axs[0, 0].set_title(r'Graph 1 ' '\n'  '$\hat{p} = $' f'{estimates1[-1][1]:.2f}' ", " '$\hat{q} = $' f'{estimates1[-1][2]:.2f}')
    axs[0, 0].plot(estimates1[:, 0], [true_values1[0]] * len(estimates1[:, 0]), c = 'C0', linestyle = 'dotted', label = "$p$", linewidth=2.5)
    axs[0, 0].plot(estimates1[:, 0], estimates1[:, 1], label = "$\hat{p}$")
    axs[0, 0].plot(estimates1[:, 0], [true_values1[1]] * len(estimates1[:, 0]), c = 'C1', linestyle = 'dotted', label = "$q$", linewidth=2.5)
    axs[0, 0].plot(estimates1[:, 0], estimates1[:, 2], label = "$\hat{q}$")
    axs[0, 0].legend(loc='center left', bbox_to_anchor=(-0.6, 0.5))
    axs[1, 0].set_title('$\hat{\gamma}_{e, z_u} = $' f'{estimates1[-1][5]:.2f}' ", " '$\hat{\gamma}_{e, \\bar{z}_u} = $' f'{estimates1[-1][6]:.2f}')
    axs[1, 0].plot(estimates1[:, 0], [true_values1[4]] * len(estimates1[:, 0]), c = 'C0', linestyle = 'dotted', label = "$\gamma_{e, z_u}$", linewidth=2.5)
    axs[1, 0].plot(estimates1[:, 0], estimates1[:, 5], label = "$\hat{\gamma}_{e, z_u}$")
    axs[1, 0].plot(estimates1[:, 0], [true_values1[5]] * len(estimates1[:, 0]), c = 'C1', linestyle = 'dotted', label = "$\gamma_{e, \\bar{z}_u}$", linewidth=2.5)
    axs[1, 0].plot(estimates1[:, 0], estimates1[:, 6], label = "$\hat{\gamma}_{e, \\bar{z}_u}$")
    axs[1, 0].legend(loc='center left', bbox_to_anchor=(-0.6, 0.5))
    axs[2, 0].set_title('$\hat{\gamma}_{n, z_u} = $' f'{estimates1[-1][3]:.2f}' ", " '$\hat{\gamma}_{n, \\bar{z}_u} = $' f'{estimates1[-1][4]:.2f}')
    axs[2, 0].plot(estimates1[:, 0], [true_values1[2]] * len(estimates1[:, 0]), c = 'C0', linestyle = 'dotted', label = "$\gamma_{n, z_u}$", linewidth=2.5)
    axs[2, 0].plot(estimates1[:, 0], estimates1[:, 3], label = "$\hat{\gamma}_{n, z_u}$")
    axs[2, 0].plot(estimates1[:, 0], [true_values1[3]] * len(estimates1[:, 0]), c = 'C1', linestyle = 'dotted', label = '$\gamma_{n, \\bar{z}_u}$', linewidth=2.5)
    axs[2, 0].plot(estimates1[:, 0], estimates1[:, 4], label = "$\hat{\gamma}_{n, \\bar{z}_u}$")
    axs[2, 0].legend(loc='center left', bbox_to_anchor=(-0.6, 0.5))

    axs[0, 1].set_title(r'Graph 2 ' '\n'  '$\hat{p} = $' f'{estimates2[-1][1]:.2f}' ", " '$\hat{q} = $' f'{estimates2[-1][2]:.2f}')
    axs[0, 1].plot(estimates2[:, 0], [true_values2[0]] * len(estimates2[:, 0]), c = 'C0', linestyle = 'dotted', linewidth=2.5)
    axs[0, 1].plot(estimates2[:, 0], [true_values2[1]] * len(estimates2[:, 0]), c = 'C1', linestyle = 'dotted', linewidth=2.5)
    axs[0, 1].plot(estimates2[:, 0], estimates2[:, 1])
    axs[0, 1].plot(estimates2[:, 0], estimates2[:, 2])
    axs[1, 1].set_title('$\hat{\gamma}_{e, z_u} = $' f'{estimates2[-1][5]:.2f}' ", " '$\hat{\gamma}_{e, \\bar{z}_u} = $' f'{estimates2[-1][6]:.2f}')
    axs[1, 1].plot(estimates2[:, 0], [true_values2[4]] * len(estimates2[:, 0]), c = 'C0', linestyle = 'dotted', linewidth=2.5)
    axs[1, 1].plot(estimates2[:, 0], [true_values2[5]] * len(estimates2[:, 0]), c = 'C1', linestyle = 'dotted', linewidth=2.5)
    axs[1, 1].plot(estimates2[:, 0], estimates2[:, 5])
    axs[1, 1].plot(estimates2[:, 0], estimates2[:, 6])
    axs[2, 1].set_title('$\hat{\gamma}_{n, z_u} = $' f'{estimates2[-1][3]:.2f}' ", " '$\hat{\gamma}_{n, \\bar{z}_u} = $' f'{estimates2[-1][4]:.2f}')
    axs[2, 1].plot(estimates2[:, 0], [true_values2[2]] * len(estimates2[:, 0]), c = 'C0', linestyle = 'dotted', linewidth=2.5)
    axs[2, 1].plot(estimates2[:, 0], [true_values2[3]] * len(estimates2[:, 0]), c = 'C1', linestyle = 'dotted', linewidth=2.5)
    axs[2, 1].plot(estimates2[:, 0], estimates2[:, 3])
    axs[2, 1].plot(estimates2[:, 0], estimates2[:, 4])

    axs[0, 2].set_title(r'Graph 3 ' '\n'  '$\hat{p} = $' f'{estimates3[-1][1]:.2f}' ", " '$\hat{q} = $' f'{estimates3[-1][2]:.2f}')
    axs[0, 2].plot(estimates3[:, 0], [true_values3[0]] * len(estimates3[:, 0]), c = 'C0', linestyle = 'dotted', linewidth=2.5)
    axs[0, 2].plot(estimates3[:, 0], [true_values3[1]] * len(estimates3[:, 0]), c = 'C1', linestyle = 'dotted', linewidth=2.5)
    axs[0, 2].plot(estimates3[:, 0], estimates3[:, 1])
    axs[0, 2].plot(estimates3[:, 0], estimates3[:, 2])
    axs[1, 2].set_title('$\hat{\gamma}_{e, z_u} = $' f'{estimates3[-1][5]:.2f}' ", " '$\hat{\gamma}_{e, \\bar{z}_u} = $' f'{estimates3[-1][6]:.2f}')
    axs[1, 2].plot(estimates3[:, 0], [true_values3[4]] * len(estimates3[:, 0]), c = 'C0', linestyle = 'dotted', linewidth=2.5)
    axs[1, 2].plot(estimates3[:, 0], [true_values3[5]] * len(estimates3[:, 0]), c = 'C1', linestyle = 'dotted', linewidth=2.5)
    axs[1, 2].plot(estimates3[:, 0], estimates3[:, 5])
    axs[1, 2].plot(estimates3[:, 0], estimates3[:, 6])
    axs[2, 2].set_title('$\hat{\gamma}_{n, z_u} = $' f'{estimates3[-1][3]:.2f}' ", " '$\hat{\gamma}_{n, \\bar{z}_u} = $' f'{estimates3[-1][4]:.2f}')
    axs[2, 2].plot(estimates3[:, 0], [true_values3[2]] * len(estimates3[:, 0]), c = 'C0', linestyle = 'dotted', linewidth=2.5)
    axs[2, 2].plot(estimates3[:, 0], [true_values3[3]] * len(estimates3[:, 0]), c = 'C1', linestyle = 'dotted', linewidth=2.5)
    axs[2, 2].plot(estimates3[:, 0], estimates3[:, 3])
    axs[2, 2].plot(estimates3[:, 0], estimates3[:, 4])

    axs[0, 3].set_title(r'Graph 4 ' '\n'  '$\hat{p} = $' f'{estimates4[-1][1]:.2f}' ", " '$\hat{q} = $' f'{estimates4[-1][2]:.2f}')
    axs[0, 3].plot(estimates4[:, 0], [true_values4[0]] * len(estimates4[:, 0]), c = 'C0', linestyle = 'dotted', linewidth=2.5)
    axs[0, 3].plot(estimates4[:, 0], [true_values4[1]] * len(estimates4[:, 0]), c = 'C1', linestyle = 'dotted', linewidth=2.5)
    axs[0, 3].plot(estimates4[:, 0], estimates4[:, 1])
    axs[0, 3].plot(estimates4[:, 0], estimates4[:, 2])
    axs[1, 3].set_title('$\hat{\gamma}_{e, z_u} = $' f'{estimates4[-1][5]:.2f}' ", " '$\hat{\gamma}_{e, \\bar{z}_u} = $' f'{estimates4[-1][6]:.2f}')
    axs[1, 3].plot(estimates4[:, 0], [true_values4[4]] * len(estimates4[:, 0]), c = 'C0', linestyle = 'dotted', linewidth=2.5)
    axs[1, 3].plot(estimates4[:, 0], [true_values4[5]] * len(estimates4[:, 0]), c = 'C1', linestyle = 'dotted', linewidth=2.5)
    axs[1, 3].plot(estimates4[:, 0], estimates4[:, 5])
    axs[1, 3].plot(estimates4[:, 0], estimates4[:, 6])
    axs[2, 3].set_title('$\hat{\gamma}_{n, z_u} = $' f'{estimates4[-1][3]:.2f}' ", " '$\hat{\gamma}_{n, \\bar{z}_u} = $' f'{estimates4[-1][4]:.2f}')
    axs[2, 3].plot(estimates4[:, 0], [true_values4[2]] * len(estimates4[:, 0]), c = 'C0', linestyle = 'dotted', linewidth=2.5)
    axs[2, 3].plot(estimates4[:, 0], [true_values4[3]] * len(estimates4[:, 0]), c = 'C1', linestyle = 'dotted', linewidth=2.5)
    axs[2, 3].plot(estimates4[:, 0], estimates4[:, 3])
    axs[2, 3].plot(estimates4[:, 0], estimates4[:, 4])

    axs[3, 0].set_ylim(min(min(liks1), min(liks2), min(liks3), min(liks4)), max(max(liks1), max(liks2), max(liks3), max(liks4)) + 100)
    axs[3, 1].set_ylim(min(min(liks1), min(liks2), min(liks3), min(liks4)), max(max(liks1), max(liks2), max(liks3), max(liks4)) + 100)
    axs[3, 2].set_ylim(min(min(liks1), min(liks2), min(liks3), min(liks4)), max(max(liks1), max(liks2), max(liks3), max(liks4)) + 100)
    axs[3, 3].set_ylim(min(min(liks1), min(liks2), min(liks3), min(liks4)), max(max(liks1), max(liks2), max(liks3), max(liks4)) + 100)
    axs[3, 0].plot(estimates1[:, 0], liks1, c = 'C2')
    axs[3, 1].plot(estimates2[:, 0], liks2, c = 'C2')
    axs[3, 2].plot(estimates3[:, 0], liks3, c = 'C2')
    axs[3, 3].plot(estimates4[:, 0], liks4, c = 'C2')
    axs[3, 0].set_ylabel("log marginal \n likelihood")
    axs[3, 0].set_xlabel("iterations")
    axs[3, 1].set_xlabel("iterations")
    axs[3, 2].set_xlabel("iterations")
    axs[3, 3].set_xlabel("iterations")

    fig.savefig('fig/sem_convergence_with_likelihood.png', dpi=300, bbox_inches = "tight")

    return plt

estimates1 = []
with open('throughput/graph1_sem_ests.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        estimates1.append(row)
estimates1 = np.array(estimates1).astype(float)
liks1 = []
with open('throughput/graph1_sem_liks.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        liks1.append(row)
liks1 = np.array(liks1).astype(float)

estimates2 = []
with open('throughput/graph2_sem_ests.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        estimates2.append(row)
estimates2 = np.array(estimates2).astype(float)
liks2 = []
with open('throughput/graph2_sem_liks.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        liks2.append(row)
liks2 = np.array(liks2).astype(float)

estimates3 = []
with open('throughput/graph3_sem_ests.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        estimates3.append(row)
estimates3 = np.array(estimates3).astype(float)
liks3 = []
with open('throughput/graph3_sem_liks.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        liks3.append(row)
liks3 = np.array(liks3).astype(float)

estimates4 = []
with open('throughput/graph4_sem_ests.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        estimates4.append(row)
estimates4 = np.array(estimates4).astype(float)
liks4 = []
with open('throughput/graph4_sem_liks.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        liks4.append(row)
liks4 = np.array(liks4).astype(float)

g1_true_theta = [0.9, 0.1, 0.75, 0.25, 0.75, 0.25]
g2_true_theta = [0.6, 0.4, 0.75, 0.25, 0.75, 0.25]
g3_true_theta = [0.1, 0.9, 0.9, 0.1, 0.9, 0.1]
g4_true_theta = [0.6, 0.4, 0.9, 0.1, 0.9, 0.1]

viz_with_likelihood(g1_true_theta, estimates1, liks1, g2_true_theta, estimates2, liks2, g3_true_theta, estimates3, liks3, g4_true_theta, estimates4, liks4)

## Plotting without likelihood

def viz_without_likelihood(true_values1, estimates1, true_values2, estimates2, true_values3, estimates3, true_values4, estimates4):
    sns.set_style("whitegrid")
    sns.set_palette("Dark2")
    
    fig, axs = plt.subplots(3, 4, sharex = True, sharey = True)
    fig.set_figwidth(20)
    fig.set_figheight(7.5)

    plt.subplots_adjust(hspace = 0.4)
    plt.setp(axs, ylim=(0, 1.1))

    plt.rcParams.update({'font.size': 17})
    plt.rc('xtick', labelsize=13) 
    plt.rc('ytick', labelsize=13) 

    axs[0, 0].set_title(r'Graph 1 ' '\n'  '$\hat{p} = $' f'{estimates1[-1][1]:.2f}' ", " '$\hat{q} = $' f'{estimates1[-1][2]:.2f}')
    axs[0, 0].plot(estimates1[:, 0], [true_values1[0]] * len(estimates1[:, 0]), c = 'C0', linestyle = 'dotted', label = "$p$", linewidth=2.5)
    axs[0, 0].plot(estimates1[:, 0], estimates1[:, 1], label = "$\hat{p}$")
    axs[0, 0].plot(estimates1[:, 0], [true_values1[1]] * len(estimates1[:, 0]), c = 'C1', linestyle = 'dotted', label = "$q$", linewidth=2.5)
    axs[0, 0].plot(estimates1[:, 0], estimates1[:, 2], label = "$\hat{q}$")
    axs[0, 0].legend(loc='center left', bbox_to_anchor=(-0.6, 0.5))
    axs[1, 0].set_title('$\hat{\gamma}_{e, z_u} = $' f'{estimates1[-1][5]:.2f}' ", " '$\hat{\gamma}_{e, \\bar{z}_u} = $' f'{estimates1[-1][6]:.2f}')
    axs[1, 0].plot(estimates1[:, 0], [true_values1[4]] * len(estimates1[:, 0]), c = 'C0', linestyle = 'dotted', label = "$\gamma_{e, z_u}$", linewidth=2.5)
    axs[1, 0].plot(estimates1[:, 0], estimates1[:, 5], label = "$\hat{\gamma}_{e, z_u}$")
    axs[1, 0].plot(estimates1[:, 0], [true_values1[5]] * len(estimates1[:, 0]), c = 'C1', linestyle = 'dotted', label = "$\gamma_{e, \\bar{z}_u}$", linewidth=2.5)
    axs[1, 0].plot(estimates1[:, 0], estimates1[:, 6], label = "$\hat{\gamma}_{e, \\bar{z}_u}$")
    axs[1, 0].legend(loc='center left', bbox_to_anchor=(-0.6, 0.5))
    axs[2, 0].set_title('$\hat{\gamma}_{n, z_u} = $' f'{estimates1[-1][3]:.2f}' ", " '$\hat{\gamma}_{n, \\bar{z}_u} = $' f'{estimates1[-1][4]:.2f}')
    axs[2, 0].plot(estimates1[:, 0], [true_values1[2]] * len(estimates1[:, 0]), c = 'C0', linestyle = 'dotted', label = "$\gamma_{n, z_u}$", linewidth=2.5)
    axs[2, 0].plot(estimates1[:, 0], estimates1[:, 3], label = "$\hat{\gamma}_{n, z_u}$")
    axs[2, 0].plot(estimates1[:, 0], [true_values1[3]] * len(estimates1[:, 0]), c = 'C1', linestyle = 'dotted', label = '$\gamma_{n, \\bar{z}_u}$', linewidth=2.5)
    axs[2, 0].plot(estimates1[:, 0], estimates1[:, 4], label = "$\hat{\gamma}_{n, \\bar{z}_u}$")
    axs[2, 0].legend(loc='center left', bbox_to_anchor=(-0.6, 0.5))

    axs[0, 1].set_title(r'Graph 2 ' '\n'  '$\hat{p} = $' f'{estimates2[-1][1]:.2f}' ", " '$\hat{q} = $' f'{estimates2[-1][2]:.2f}')
    axs[0, 1].plot(estimates2[:, 0], [true_values2[0]] * len(estimates2[:, 0]), c = 'C0', linestyle = 'dotted', linewidth=2.5)
    axs[0, 1].plot(estimates2[:, 0], [true_values2[1]] * len(estimates2[:, 0]), c = 'C1', linestyle = 'dotted', linewidth=2.5)
    axs[0, 1].plot(estimates2[:, 0], estimates2[:, 1])
    axs[0, 1].plot(estimates2[:, 0], estimates2[:, 2])
    axs[1, 1].set_title('$\hat{\gamma}_{e, z_u} = $' f'{estimates2[-1][5]:.2f}' ", " '$\hat{\gamma}_{e, \\bar{z}_u} = $' f'{estimates2[-1][6]:.2f}')
    axs[1, 1].plot(estimates2[:, 0], [true_values2[4]] * len(estimates2[:, 0]), c = 'C0', linestyle = 'dotted', linewidth=2.5)
    axs[1, 1].plot(estimates2[:, 0], [true_values2[5]] * len(estimates2[:, 0]), c = 'C1', linestyle = 'dotted', linewidth=2.5)
    axs[1, 1].plot(estimates2[:, 0], estimates2[:, 5])
    axs[1, 1].plot(estimates2[:, 0], estimates2[:, 6])
    axs[2, 1].set_title('$\hat{\gamma}_{n, z_u} = $' f'{estimates2[-1][3]:.2f}' ", " '$\hat{\gamma}_{n, \\bar{z}_u} = $' f'{estimates2[-1][4]:.2f}')
    axs[2, 1].plot(estimates2[:, 0], [true_values2[2]] * len(estimates2[:, 0]), c = 'C0', linestyle = 'dotted', linewidth=2.5)
    axs[2, 1].plot(estimates2[:, 0], [true_values2[3]] * len(estimates2[:, 0]), c = 'C1', linestyle = 'dotted', linewidth=2.5)
    axs[2, 1].plot(estimates2[:, 0], estimates2[:, 3])
    axs[2, 1].plot(estimates2[:, 0], estimates2[:, 4])

    axs[0, 2].set_title(r'Graph 3 ' '\n'  '$\hat{p} = $' f'{estimates3[-1][1]:.2f}' ", " '$\hat{q} = $' f'{estimates3[-1][2]:.2f}')
    axs[0, 2].plot(estimates3[:, 0], [true_values3[0]] * len(estimates3[:, 0]), c = 'C0', linestyle = 'dotted', linewidth=2.5)
    axs[0, 2].plot(estimates3[:, 0], [true_values3[1]] * len(estimates3[:, 0]), c = 'C1', linestyle = 'dotted', linewidth=2.5)
    axs[0, 2].plot(estimates3[:, 0], estimates3[:, 1])
    axs[0, 2].plot(estimates3[:, 0], estimates3[:, 2])
    axs[1, 2].set_title('$\hat{\gamma}_{e, z_u} = $' f'{estimates3[-1][5]:.2f}' ", " '$\hat{\gamma}_{e, \\bar{z}_u} = $' f'{estimates3[-1][6]:.2f}')
    axs[1, 2].plot(estimates3[:, 0], [true_values3[4]] * len(estimates3[:, 0]), c = 'C0', linestyle = 'dotted', linewidth=2.5)
    axs[1, 2].plot(estimates3[:, 0], [true_values3[5]] * len(estimates3[:, 0]), c = 'C1', linestyle = 'dotted', linewidth=2.5)
    axs[1, 2].plot(estimates3[:, 0], estimates3[:, 5])
    axs[1, 2].plot(estimates3[:, 0], estimates3[:, 6])
    axs[2, 2].set_title('$\hat{\gamma}_{n, z_u} = $' f'{estimates3[-1][3]:.2f}' ", " '$\hat{\gamma}_{n, \\bar{z}_u} = $' f'{estimates3[-1][4]:.2f}')
    axs[2, 2].plot(estimates3[:, 0], [true_values3[2]] * len(estimates3[:, 0]), c = 'C0', linestyle = 'dotted', linewidth=2.5)
    axs[2, 2].plot(estimates3[:, 0], [true_values3[3]] * len(estimates3[:, 0]), c = 'C1', linestyle = 'dotted', linewidth=2.5)
    axs[2, 2].plot(estimates3[:, 0], estimates3[:, 3])
    axs[2, 2].plot(estimates3[:, 0], estimates3[:, 4])

    axs[0, 3].set_title(r'Graph 4 ' '\n'  '$\hat{p} = $' f'{estimates4[-1][1]:.2f}' ", " '$\hat{q} = $' f'{estimates4[-1][2]:.2f}')
    axs[0, 3].plot(estimates4[:, 0], [true_values4[0]] * len(estimates4[:, 0]), c = 'C0', linestyle = 'dotted', linewidth=2.5)
    axs[0, 3].plot(estimates4[:, 0], [true_values4[1]] * len(estimates4[:, 0]), c = 'C1', linestyle = 'dotted', linewidth=2.5)
    axs[0, 3].plot(estimates4[:, 0], estimates4[:, 1])
    axs[0, 3].plot(estimates4[:, 0], estimates4[:, 2])
    axs[1, 3].set_title('$\hat{\gamma}_{e, z_u} = $' f'{estimates4[-1][5]:.2f}' ", " '$\hat{\gamma}_{e, \\bar{z}_u} = $' f'{estimates4[-1][6]:.2f}')
    axs[1, 3].plot(estimates4[:, 0], [true_values4[4]] * len(estimates4[:, 0]), c = 'C0', linestyle = 'dotted', linewidth=2.5)
    axs[1, 3].plot(estimates4[:, 0], [true_values4[5]] * len(estimates4[:, 0]), c = 'C1', linestyle = 'dotted', linewidth=2.5)
    axs[1, 3].plot(estimates4[:, 0], estimates4[:, 5])
    axs[1, 3].plot(estimates4[:, 0], estimates4[:, 6])
    axs[2, 3].set_title('$\hat{\gamma}_{n, z_u} = $' f'{estimates4[-1][3]:.2f}' ", " '$\hat{\gamma}_{n, \\bar{z}_u} = $' f'{estimates4[-1][4]:.2f}')
    axs[2, 3].plot(estimates4[:, 0], [true_values4[2]] * len(estimates4[:, 0]), c = 'C0', linestyle = 'dotted', linewidth=2.5)
    axs[2, 3].plot(estimates4[:, 0], [true_values4[3]] * len(estimates4[:, 0]), c = 'C1', linestyle = 'dotted', linewidth=2.5)
    axs[2, 3].plot(estimates4[:, 0], estimates4[:, 3])
    axs[2, 3].plot(estimates4[:, 0], estimates4[:, 4])


    axs[2, 0].set_xlabel("iterations")
    axs[2, 1].set_xlabel("iterations")
    axs[2, 2].set_xlabel("iterations")
    axs[2, 3].set_xlabel("iterations")

    fig.savefig('fig/sem_convergence_without_likelihood.png', dpi=300, bbox_inches = "tight")

    return plt

estimates1_extended = []
with open('throughput/graph1_sem_ests_extended.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        estimates1_extended.append(row)
estimates1_extended = np.array(estimates1_extended).astype(float)

estimates2_extended = []
with open('throughput/graph2_sem_ests_extended.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        estimates2_extended.append(row)
estimates2_extended = np.array(estimates2_extended).astype(float)

estimates3_extended = []
with open('throughput/graph3_sem_ests_extended.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        estimates3_extended.append(row)
estimates3_extended = np.array(estimates3_extended).astype(float)

estimates4_extended = []
with open('throughput/graph4_sem_ests_extended.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        estimates4_extended.append(row)
estimates4_extended = np.array(estimates4_extended).astype(float)

g1_true_theta = [0.9, 0.1, 0.75, 0.25, 0.75, 0.25]
g2_true_theta = [0.6, 0.4, 0.75, 0.25, 0.75, 0.25]
g3_true_theta = [0.1, 0.9, 0.9, 0.1, 0.9, 0.1]
g4_true_theta = [0.6, 0.4, 0.9, 0.1, 0.9, 0.1]

viz_with_likelihood(g1_true_theta, estimates1, liks1, g2_true_theta, estimates2, liks2, g3_true_theta, estimates3, liks3, g4_true_theta, estimates4, liks4)
viz_without_likelihood(g1_true_theta, estimates1_extended, g2_true_theta, estimates2_extended, g3_true_theta, estimates3_extended, g4_true_theta, estimates4_extended)
