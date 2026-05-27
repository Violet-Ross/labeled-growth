import xgi
from src.algorithms.sem import sem_functions
import numpy as np
import csv


sem = sem_functions()
s_intial = np.array([1, 2, 1, 2, 0.5, 0.5, 0.5, 0.5])
g1_true_theta = [0.9, 0.1, 0.75, 0.25, 0.75, 0.25]
g2_true_theta = [0.6, 0.4, 0.75, 0.25, 0.75, 0.25]
g3_true_theta = [0.1, 0.9, 2, 2, 0.75, 0.25]
g4_true_theta = [0.6, 0.4, 2, 1.8, 0.75, 0.25]

initial_rate = 0.01
constant = 0.001
no_likelihood_iteration_limit = 1000
with_likelihood_iteration_limit = 8000

print("Graph 1")
GH1 = sem.generate_hypergraph(g1_true_theta, 500)
estimates1, likelihoods1 = sem.SEM_with_likelihood(GH1, s_intial, no_likelihood_iteration_limit, initial_rate, constant)
with open('throughput/graph1_sem_ests.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(estimates1)
np.savetxt('throughput/graph1_sem_liks.csv', likelihoods1, delimiter=',') 

print("Graph 2")
GH2 = sem.generate_hypergraph(g2_true_theta, 500)
estimates2, likelihoods2 = sem.SEM_with_likelihood(GH2, s_intial, no_likelihood_iteration_limit, initial_rate, constant)
with open('throughput/graph2_sem_ests.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(estimates2)
np.savetxt('throughput/graph2_sem_liks.csv', likelihoods2, delimiter=',') 

print("Graph 3")
GH3 = sem.generate_hypergraph(g3_true_theta, 500)
estimates3, likelihoods3 = sem.SEM_with_likelihood(GH3, s_intial, no_likelihood_iteration_limit, initial_rate, constant)
with open('throughput/graph3_sem_ests.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(estimates3)
np.savetxt('throughput/graph3_sem_liks.csv', likelihoods3, delimiter=',') 

print("Graph 4")
GH4 = sem.generate_hypergraph(g4_true_theta, 500)
estimates4, likelihoods4 = sem.SEM_with_likelihood(GH4, s_intial, no_likelihood_iteration_limit, initial_rate, constant)
with open('throughput/graph4_sem_ests.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(estimates4)
np.savetxt('throughput/graph4_sem_liks.csv', likelihoods4, delimiter=',') 

print("Graph 1")
GH1 = sem.generate_hypergraph(g1_true_theta, 2000)
estimates1 = sem.SEM_without_likelihood(GH1, s_intial, with_likelihood_iteration_limit, initial_rate, constant)
with open('throughput/graph1_sem_ests_extended.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(estimates1)
print("Graph 2")
GH2 = sem.generate_hypergraph(g2_true_theta, 2000)
estimates2 = sem.SEM_without_likelihood(GH2, s_intial, with_likelihood_iteration_limit, initial_rate, constant)
with open('throughput/graph2_sem_ests_extended.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(estimates2)
print("Graph 3")
GH3 = sem.generate_hypergraph(g3_true_theta, 2000)
estimates3 = sem.SEM_without_likelihood(GH3, s_intial, with_likelihood_iteration_limit, initial_rate, constant)
with open('throughput/graph3_sem_ests_extended.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(estimates3)
print("Graph 4")
GH4 = sem.generate_hypergraph(g4_true_theta, 2000)
estimates4 = sem.SEM_without_likelihood(GH4, s_intial, with_likelihood_iteration_limit, initial_rate, constant)
with open('throughput/graph4_sem_ests_extended.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(estimates4)
