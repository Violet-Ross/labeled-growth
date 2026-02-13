import poisson_hypergraph
import xgi
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns

def assm1_viz(data_9_1, data_6_4, data_1_9):
    sns.set_style("whitegrid")
    sns.set_palette("Dark2")
    plt.rcParams.update({'font.size': 20})
    plt.rc('xtick', labelsize=12) 
    plt.rc('ytick', labelsize=12) 

    
    fig, axs = plt.subplots(6, 3, sharex = True)

    # fig.suptitle("Assumption 1")
    axs[0, 0].set_ylim(0,10)
    axs[0, 0].set_title('p = 0.9, q = 0.1')
    axs[0, 0].plot(data_9_1[:, 0], data_9_1[:, 1], c = 'C0', label = "$\\langle s_{(0),e} \\rangle$")
    axs[0, 0].scatter(data_9_1[:, 0], data_9_1[:, 2], c = 'C1', label = "$\\langle s_{(0),h} \\rangle$")
    axs[0, 0].legend(loc='center left', bbox_to_anchor=(-0.8, 0.5))
    axs[1, 0].set_ylim(0,5)
    axs[1, 0].plot(data_9_1[:, 0], data_9_1[:, 3], c = 'C0', label = "$\\langle s_{(1),e} \\rangle$")
    axs[1, 0].scatter(data_9_1[:, 0], data_9_1[:, 4], c = 'C1', label = "$\\langle s_{(1),h} \\rangle$")
    axs[1, 0].legend(loc='center left', bbox_to_anchor=(-0.8, 0.5))
    axs[2, 0].set_ylim(0,1)
    axs[2, 0].plot(data_9_1[:, 0], data_9_1[:, 5], c = 'C0', label = "$\\langle \\frac{s_{(0),e}}{s_e} \\rangle$")
    axs[2, 0].scatter(data_9_1[:, 0], data_9_1[:, 6], c = 'C1', label = "$\\langle \\frac{s_{(0),h}}{s_h} \\rangle$")
    axs[2, 0].legend(loc='center left', bbox_to_anchor=(-0.8, 0.5))
    axs[3, 0].plot(data_9_1[:, 0], data_9_1[:, 7], c = 'C0', label = "$\\langle \\frac{s_{(1),e}}{s_e} \\rangle$")
    axs[3, 0].set_ylim(0,1)
    axs[3, 0].scatter(data_9_1[:, 0], data_9_1[:, 8], c = 'C1', label = "$\\langle \\frac{s_{(1),h}}{s_h} \\rangle$")
    axs[3, 0].legend(loc='center left', bbox_to_anchor=(-0.8, 0.5))
    axs[4, 0].set_ylim(0,10)
    axs[4, 0].plot(data_9_1[:, 0], data_9_1[:, 9], c = 'C0', label = "$\\langle \\frac{s_{(0),e}^2}{s_e} \\rangle$")
    axs[4, 0].scatter(data_9_1[:, 0], data_9_1[:, 10], c = 'C1', label = "$\\langle \\frac{s_{(0),h}^2}{s_h} \\rangle$")
    axs[4, 0].legend(loc='center left', bbox_to_anchor=(-0.8, 0.5))
    axs[5, 0].plot(data_9_1[:, 0], data_9_1[:, 11], c = 'C0', label = "$\\langle \\frac{s_{(1),e}^2}{s_e} \\rangle$")
    axs[5, 0].scatter(data_9_1[:, 0], data_9_1[:, 12], c = 'C1', label = "$\\langle \\frac{s_{(1),h}^2}{s_h} \\rangle$")
    axs[5, 0].set_ylim(0,4)
    axs[5, 0].set_xlabel("$\\beta$")
    axs[5, 0].legend(loc='center left', bbox_to_anchor=(-0.8, 0.5))

    axs[0, 1].set_ylim(0,10)
    axs[0, 1].set_title('p = 0.6, q = 0.4')
    axs[0, 1].plot(data_6_4[:, 0], data_6_4[:, 1], c = 'C0')
    axs[0, 1].scatter(data_6_4[:, 0], data_6_4[:, 2], c = 'C1')
    axs[1, 1].set_ylim(0,5)
    axs[1, 1].plot(data_6_4[:, 0], data_6_4[:, 3], c = 'C0')
    axs[1, 1].scatter(data_6_4[:, 0], data_6_4[:, 4], c = 'C1')
    axs[2, 1].set_ylim(0,1)
    axs[2, 1].plot(data_6_4[:, 0], data_6_4[:, 5], c = 'C0')
    axs[2, 1].scatter(data_6_4[:, 0], data_6_4[:, 6], c = 'C1')
    axs[3, 1].set_ylim(0,1)
    axs[3, 1].plot(data_6_4[:, 0], data_6_4[:, 7], c = 'C0')
    axs[3, 1].scatter(data_6_4[:, 0], data_6_4[:, 8], c = 'C1')
    axs[4, 1].set_ylim(0,10)
    axs[4, 1].plot(data_6_4[:, 0], data_6_4[:, 9], c = 'C0')
    axs[4, 1].scatter(data_6_4[:, 0], data_6_4[:, 10], c = 'C1')
    axs[5, 1].plot(data_6_4[:, 0], data_6_4[:, 11], c = 'C0')
    axs[5, 1].scatter(data_6_4[:, 0], data_6_4[:, 12], c = 'C1')
    axs[5, 1].set_ylim(0,4)
    axs[5, 1].set_xlabel("$\\beta$")
    
    axs[0, 2].set_ylim(0,10)
    axs[0, 2].set_title('p = 0.1, q = 0.9')
    axs[0, 2].plot(data_1_9[:, 0], data_1_9[:, 1], c = 'C0')
    axs[0, 2].scatter(data_1_9[:, 0], data_1_9[:, 2], c = 'C1')
    axs[1, 2].set_ylim(0,5)
    axs[1, 2].plot(data_1_9[:, 0], data_1_9[:, 3], c = 'C0')
    axs[1, 2].scatter(data_1_9[:, 0], data_1_9[:, 4], c = 'C1')
    axs[2, 2].set_ylim(0,1)
    axs[2, 2].plot(data_1_9[:, 0], data_1_9[:, 5], c = 'C0')
    axs[2, 2].scatter(data_1_9[:, 0], data_1_9[:, 6], c = 'C1')
    axs[3, 2].set_ylim(0,1)
    axs[3, 2].plot(data_1_9[:, 0], data_1_9[:, 7], c = 'C0')
    axs[3, 2].scatter(data_1_9[:, 0], data_1_9[:, 8], c = 'C1')
    axs[4, 2].set_ylim(0,10)
    axs[4, 2].plot(data_1_9[:, 0], data_1_9[:, 9], c = 'C0')
    axs[4, 2].scatter(data_1_9[:, 0], data_1_9[:, 10], c = 'C1')
    axs[5, 2].plot(data_1_9[:, 0], data_1_9[:, 11], c = 'C0')
    axs[5, 2].scatter(data_1_9[:, 0], data_1_9[:, 12], c = 'C1')
    axs[5, 2].set_ylim(0,4)
    axs[5, 2].set_xlabel("$\\beta$")

    fig.set_size_inches(14, 19)
    # plt.setp(axs, ylim=(0, 1))
    fig.savefig('fig/assm1.png', dpi=300, bbox_inches='tight')
    return plt

def assm2_viz(data_9_1, data_6_4, data_1_9):
    sns.set_style("whitegrid")
    sns.set_palette("Dark2")
    
    fig, axs = plt.subplots(2, 3, sharex = True, sharey = True)
    fig.set_figwidth(20)
    fig.set_figheight(7)

    plt.rcParams.update({'font.size': 20})
    plt.rc('xtick', labelsize=12) 
    plt.rc('ytick', labelsize=12) 
    # fig.suptitle("Assumption 1")
    axs[0, 0].set_title('p = 0.9, q = 0.1')
    axs[0, 0].plot(data_9_1[:, 0], data_9_1[:, 1], c = 'C0', label = "$\\left \\langle \\frac{s_0}{s} \\right \\rangle$")
    axs[0, 0].scatter(data_9_1[:, 0], data_9_1[:, 2], c = 'C1', label = "$\\frac{\\langle s_0 \\rangle }{\\langle s \\rangle}$")
    axs[0, 0].legend(loc='center left', bbox_to_anchor=(-0.65, 0.5), fontsize=17)
    axs[1, 0].plot(data_9_1[:, 0], data_9_1[:, 3], c = 'C0', label = "$\\left \\langle \\frac{s_1}{s} \\right \\rangle$")
    axs[1, 0].scatter(data_9_1[:, 0], data_9_1[:, 4], c = 'C1', label = "$\\frac{\\langle s_1 \\rangle }{\\langle s \\rangle}$")
    axs[1, 0].set_xlabel("$\\beta$")
    axs[1, 0].legend(loc='center left', bbox_to_anchor=(-0.65, 0.5), fontsize=17)
    axs[0, 1].set_title('p = 0.6, q = 0.4')
    axs[0, 1].plot(data_6_4[:, 0], data_6_4[:, 1], c = 'C0')
    axs[0, 1].scatter(data_6_4[:, 0], data_6_4[:, 2], c = 'C1')
    axs[1, 1].plot(data_6_4[:, 0], data_6_4[:, 3], c = 'C0')
    axs[1, 1].scatter(data_6_4[:, 0], data_6_4[:, 4], c = 'C1')
    axs[1, 1].set_xlabel("$\\beta$")
    axs[0, 2].set_title('p = 0.1, q = 0.9')
    axs[0, 2].plot(data_1_9[:, 0], data_1_9[:, 1], c = 'C0')
    axs[0, 2].scatter(data_1_9[:, 0], data_1_9[:, 2], c = 'C1')
    axs[1, 2].plot(data_1_9[:, 0], data_1_9[:, 3], c = 'C0')
    axs[1, 2].scatter(data_1_9[:, 0], data_1_9[:, 4], c = 'C1')
    axs[1, 2].set_xlabel("$\\beta$")
    fig.set_size_inches(14, 7)
    plt.setp(axs, ylim=(0, 1))
    fig.savefig('fig/assm2.png', dpi=300, bbox_inches='tight')
    return plt

def assm3_viz(data_9_1, data_6_4, data_1_9):
    sns.set_style("whitegrid")
    sns.set_palette("Dark2")
    
    fig, axs = plt.subplots(2, 3, sharex = True, sharey = True)
    fig.set_figwidth(20)
    fig.set_figheight(7)
    
    plt.rcParams.update({'font.size': 20})
    plt.rc('xtick', labelsize=12) 
    plt.rc('ytick', labelsize=12) 

    # fig.suptitle("Assumption 2")
    axs[0, 0].set_title('p = 0.9, q = 0.1')
    axs[0, 0].plot(data_9_1[:, 0], data_9_1[:, 5], c = 'C0', label = "$\\left \\langle \\frac{s_0^2}{s} \\right \\rangle$")
    axs[0, 0].scatter(data_9_1[:, 0], data_9_1[:, 6], c = 'C1', label = "$\\frac{\\langle s_0^2 \\rangle }{\\langle s \\rangle}$")
    axs[0, 0].legend(loc='center left', bbox_to_anchor=(-0.65, 0.5), fontsize=17)
    axs[1, 0].plot(data_9_1[:, 0], data_9_1[:, 7], c = 'C0', label = "$\\left \\langle \\frac{s_1^2}{s} \\right \\rangle$")
    axs[1, 0].scatter(data_9_1[:, 0], data_9_1[:, 8], c = 'C1', label = "$\\frac{\\langle s_1^2 \\rangle }{\\langle s \\rangle}$")
    axs[1, 0].set_xlabel("$\\beta$")
    axs[1, 0].legend(loc='center left', bbox_to_anchor=(-0.65, 0.5), fontsize=17)
    axs[0, 1].set_title('p = 0.6, q = 0.4')
    axs[0, 1].plot(data_6_4[:, 0], data_6_4[:, 5], c = 'C0')
    axs[0, 1].scatter(data_6_4[:, 0], data_6_4[:, 6], c = 'C1')
    axs[1, 1].plot(data_6_4[:, 0], data_6_4[:, 7], c = 'C0')
    axs[1, 1].scatter(data_6_4[:, 0], data_6_4[:, 8], c = 'C1')
    axs[1, 1].set_xlabel("$\\beta$")
    axs[0, 2].set_title('p = 0.1, q = 0.9')
    axs[0, 2].plot(data_1_9[:, 0], data_1_9[:, 5], c = 'C0')
    axs[0, 2].scatter(data_1_9[:, 0], data_1_9[:, 6], c = 'C1')
    axs[1, 2].plot(data_1_9[:, 0], data_1_9[:, 7], c = 'C0')
    axs[1, 2].scatter(data_1_9[:, 0], data_1_9[:, 8], c = 'C1')
    axs[1, 2].set_xlabel("$\\beta$")
    fig.set_size_inches(14, 7)
    plt.setp(axs, ylim=(0, 8))
    fig.savefig('fig/assm3.png', dpi=300, bbox_inches='tight')
    return plt

def assm4_viz(data_9_1_square, data_6_4_square, data_1_9_square):
    sns.set_style("whitegrid")
    sns.set_palette("Dark2")
    
    fig, axs = plt.subplots(2, 3, sharex = True, sharey = True)
    fig.set_figwidth(20)
    fig.set_figheight(7)

    plt.rcParams.update({'font.size': 20})
    plt.rc('xtick', labelsize=12) 
    plt.rc('ytick', labelsize=12) 

    # fig.suptitle("Assumption 2")
    axs[0, 0].set_title('p = 0.9, q = 0.1')
    axs[0, 0].plot(data_9_1_square[:, 0], data_9_1_square[:, 1], c = 'C0', label = "$\\frac{\\langle s_0^2 \\rangle}{\\langle s \\rangle}$")
    axs[0, 0].scatter(data_9_1_square[:, 0], data_9_1_square[:, 2], c = 'C1', label = "$\\frac{\\langle s_0 \\rangle^2 + \\langle s_0 \\rangle}{\\langle s \\rangle}$")
    axs[0, 0].legend(loc='center left', bbox_to_anchor=(-0.8, 0.5), fontsize=17)
    axs[1, 0].plot(data_9_1_square[:, 0], data_9_1_square[:, 3], c = 'C0', label = "$\\frac{\\langle s_1^2 \\rangle}{\\langle s \\rangle}$")
    axs[1, 0].scatter(data_9_1_square[:, 0], data_9_1_square[:, 4], c = 'C1', label = "$\\frac{\\langle s_1 \\rangle^2 + \\langle s_1 \\rangle}{\\langle s \\rangle}$")
    axs[1, 0].set_xlabel("$\\beta$")
    axs[1, 0].legend(loc='center left', bbox_to_anchor=(-0.8, 0.5), fontsize=17)
    axs[0, 1].set_title('p = 0.6, q = 0.4')
    axs[0, 1].plot(data_6_4_square[:, 0], data_6_4_square[:, 1], c = 'C0')
    axs[0, 1].scatter(data_6_4_square[:, 0], data_6_4_square[:, 2], c = 'C1')
    axs[1, 1].plot(data_6_4_square[:, 0], data_6_4_square[:, 3], c = 'C0')
    axs[1, 1].scatter(data_6_4_square[:, 0], data_6_4_square[:, 4], c = 'C1')
    axs[1, 1].set_xlabel("$\\beta$")
    axs[0, 2].set_title('p = 0.1, q = 0.9')
    axs[0, 2].plot(data_1_9_square[:, 0], data_1_9_square[:, 1], c = 'C0')
    axs[0, 2].scatter(data_1_9_square[:, 0], data_1_9_square[:, 2], c = 'C1')
    axs[1, 2].plot(data_1_9_square[:, 0], data_1_9_square[:, 3], c = 'C0')
    axs[1, 2].scatter(data_1_9_square[:, 0], data_1_9_square[:, 4], c = 'C1')
    axs[1, 2].set_xlabel("$\\beta$")
    fig.set_size_inches(14, 7)
    #plt.setp(axs, ylim=(0, 8))

    fig.savefig('fig/assm4.png', dpi=300, bbox_inches='tight')
    return plt

assm1_6_4 = []
with open('throughput/assumption1_06_04.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        assm1_6_4.append(row)
assm1_6_4 = np.array(assm1_6_4)[1:].astype(float)

assm1_9_1 = []
with open('throughput/assumption1_09_01.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        assm1_9_1.append(row)
assm1_9_1 = np.array(assm1_9_1)[1:].astype(float)

assm1_1_9 = []
with open('throughput/assumption1_01_09.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        assm1_1_9.append(row)
assm1_1_9 = np.array(assm1_1_9)[1:].astype(float)

assm2_6_4 = []
with open('throughput/assumption2_06_04.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        assm2_6_4.append(row)
assm2_6_4 = np.array(assm2_6_4)[1:].astype(float)

assm2_9_1 = []
with open('throughput/assumption2_09_01.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        assm2_9_1.append(row)
assm2_9_1 = np.array(assm2_9_1)[1:].astype(float)

assm2_1_9 = []
with open('throughput/assumption2_01_09.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        assm2_1_9.append(row)
assm2_1_9 = np.array(assm2_1_9)[1:].astype(float)

assm4_6_4 = []
with open('throughput/assumption4_06_04.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        assm4_6_4.append(row)
assm4_6_4 = np.array(assm4_6_4)[1:].astype(float)

assm4_9_1 = []
with open('throughput/assumption4_09_01.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        assm4_9_1.append(row)
assm4_9_1 = np.array(assm4_9_1)[1:].astype(float)

assm4_1_9 = []
with open('throughput/assumption4_01_09.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        assm4_1_9.append(row)
assm4_1_9 = np.array(assm4_1_9)[1:].astype(float)

assm1_viz(assm1_9_1, assm1_6_4. assm1_1_9)
assm2_viz(assm2_9_1, assm2_6_4. assm2_1_9)
assm3_viz(assm2_9_1, assm2_6_4. assm2_1_9)
assm4_viz(assm4_9_1, assm4_6_4. assm4_1_9)