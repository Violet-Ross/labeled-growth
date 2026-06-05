import xgi
from src.algorithms.sem import sem_functions
import numpy as np
import csv


sem = sem_functions()

s_intial = np.array([1, 2, 1, 2, 0.5, 0.5, 0.5, 0.5])
initial_rate = 0.01
constant = 0.001
iteration_limit = 8000

base = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

g1 = [0.9, 0.9, 0.2, 0.2, 0.2, 0.2]
g2 = [0.2, 0.2, 2, 2, 0.2, 0.2]
g3 = [0.2, 0.2, 0.2, 0.2, 2, 2]
g4 = [0.9, 0.9, 2, 2, 2, 2]
g5 = [0.4, 0.4, 0.7, 0.7, 0.7, 0.7]

g6 = [0.8, 0.4, 0.2, 0.2, 0.2, 0.2]
g7 = [0.2, 0.2, 1, 0.5, 0.2, 0.2]
g8 = [0.2, 0.2, 0.2, 0.2, 1, 0.5]
g9 = [0.6, 0.4, 1.4, 1, 0.8, 0.2]
g10 = [0.7, 0.62, 1, 0.8, 0.3, 0.25]

g11 = [0.4, 0.8, 0.2, 0.2, 0.2, 0.2]
g12 = [0.2, 0.2, 0.5, 1, 0.2, 0.2]
g13 = [0.2, 0.2, 0.2, 0.2, 0.5, 1]
g14 = [0.4, 0.6, 1, 1.4, 0.2, 0.8]
g15 = [0.62, 0.7, 0.8, 1, 0.25, 0.3]


true_thetas = [g1, g2, g3, g4, g5,
               g6, g7, g8, g9, g10,
               g11, g12, g13, g14, g15]

for i, true_theta in enumerate(true_thetas, start=1):
    print(f"Graph {i}")
    GH = sem.generate_hypergraph(true_theta, 2000)
    estimates = sem.SEM_without_likelihood(GH, s_intial, iteration_limit, initial_rate, constant)
    with open(f'throughput/graph{i}_sem_ests_extended.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(estimates)

