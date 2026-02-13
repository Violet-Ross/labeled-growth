import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import csv

analytic = []
with open('throughput/analytic_results.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        analytic.append(row)
analytic = np.array(analytic)[1:].astype(float)

simulated = []
with open('throughput/bifurcation_sim_results.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        simulated.append(row)
simulated = np.array(simulated)[1:].astype(float)

sns.set_style("whitegrid")
sns.set_palette("Dark2")
plt.scatter(analytic[:, 3], analytic[:, 0], c = 'black', s = 1)
plt.scatter(analytic[:, 3], analytic[:, 1], c = 'black', s = 1)
plt.scatter(analytic[:, 3], analytic[:, 2], c = 'black', s = 1)
plt.xlabel("$\\beta$")
plt.ylabel("mean minority/majority edge size")
plt.xlim(0,1)
plt.savefig('fig/analytic_bifucation.png', dpi=300)

sns.set_style("whitegrid")
sns.set_palette("Dark2")
plt.scatter(analytic[:, 3], analytic[:, 0], c = 'black', s = 1, label = "analytic solutions")
plt.scatter(analytic[:, 3], analytic[:, 1], c = 'black', s = 1)
plt.scatter(analytic[:, 3], analytic[:, 2], c = 'black', s = 1)
plt.scatter(simulated[:, 0], simulated[:, 1], s = 14, c = 'C0', marker = '*', label = "simulated majority")
plt.scatter(simulated[:, 0], simulated[:, 2], s = 14, c = 'C0', marker = '^', label = "simulated minority")
plt.xlabel("$\\beta$")
plt.ylabel("mean minority/majority edge size")
plt.legend()
plt.xlim(0,1)
plt.ylim(0,18)
plt.savefig('fig/simulated_bifucation.png', dpi=300)