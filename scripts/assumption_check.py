from src.poisson_hypergraph import GH
import xgi
import matplotlib.pyplot as plt
import numpy as np
import csv

def generate_data_assm2(p, q, timesteps, experiments):
    beta_vals = np.arange(0, 1, step = 0.1)
    data = [["beta", "actual k_0/k", "estimate k_0/k", "actual k_1/k", "estimate k_1/k", 
            "actual k_0^2/k", "estimate k_0^2/k", "actual k_1^2/k", "estimate k_1^2/k", "overall 1 prop"]]
    for beta in beta_vals:
        print("beta: " + str(beta))
        estimates = []
        for experiment in range(experiments):
            H = xgi.Hypergraph([[0, 1]])
            H.set_node_attributes({0 : 0, 1 : 1}, name = "label")
            growing_hypergraph = GH(H, [0, 1], p, q)

            growing_hypergraph.add_hyperedge(timesteps, beta, 1 - beta, beta, 1 - beta)
            
            k_0_list = []
            k_1_list = []
            k_list = []
            prop_0_list = []
            prop_1_list = []
            prop_0_sq_list = []
            prop_1_sq_list = []
            labels = growing_hypergraph.node_labels
            for edge in growing_hypergraph.edge_members:
                edge_labels = [labels[node] for node in edge]
                k_1_list.append(sum(edge_labels))
                k_0_list.append(len(edge_labels) - sum(edge_labels))
                k_list.append(len(edge_labels))
                prop_1_list.append(sum(edge_labels)/len(edge_labels))
                prop_0_list.append((len(edge_labels) - sum(edge_labels))/len(edge_labels))
                prop_1_sq_list.append((sum(edge_labels)**2)/len(edge_labels))
                prop_0_sq_list.append(((len(edge_labels) - sum(edge_labels))**2)/len(edge_labels))
            mean_k_0 =  sum(k_0_list)/len(k_0_list)
            mean_k_1 =  sum(k_1_list)/len(k_1_list)
            mean_sq_0 = sum([num ** 2 for num in k_0_list])/len(k_0_list)
            mean_sq_1 = sum([num ** 2 for num in k_1_list])/len(k_1_list)
            mean_k =  sum(k_list)/len(k_list)
            mean_prop_0 = sum(prop_0_list)/len(prop_0_list)
            mean_prop_1 = sum(prop_1_list)/len(prop_1_list)
            mean_prop_0_sq = sum(prop_0_sq_list)/len(prop_0_sq_list)
            mean_prop_1_sq = sum(prop_1_sq_list)/len(prop_1_sq_list)
            estimates.append([mean_prop_0, mean_k_0/mean_k, mean_prop_1, mean_k_1/mean_k,
                    mean_prop_0_sq, mean_sq_0/mean_k,   mean_prop_1_sq, mean_sq_1/mean_k, sum(labels)/len(labels)])
        estimates = np.array(estimates)
        data.append([beta, sum(estimates[:,0]) / len(estimates[:,0]), sum(estimates[:,1]) / len(estimates[:,1]), 
                     sum(estimates[:,2]) / len(estimates[:,2]), sum(estimates[:,3]) / len(estimates[:,3]), 
                     sum(estimates[:,4]) / len(estimates[:,4]), sum(estimates[:,5]) / len(estimates[:,5]),
                     sum(estimates[:,6]) / len(estimates[:,6]), sum(estimates[:,7]) / len(estimates[:,7]),
                     sum(estimates[:,8]) / len(estimates[:,8])])
    return data

def generate_data_assm4(p, q, timesteps, experiments):
    beta_vals = np.arange(0, 1, step = 0.1)
    data = [["beta","actual k_0^2/k", "estimate k_0^2/k", "actual k_1^2/k", "estimate k_1^2/k"]]
    for beta in beta_vals:
        print("beta: " + str(beta))
        estimates = []
        for experiment in range(experiments):
            H = xgi.Hypergraph([[0, 1]])
            H.set_node_attributes({0 : 0, 1 : 1}, name = "label")
            growing_hypergraph = GH(H, [0, 1], p, q)

            growing_hypergraph.add_hyperedge(timesteps, beta, 1 - beta, beta, 1 - beta)
            
            k_0_list = []
            k_1_list = []
            k_list = []
            prop_0_list = []
            prop_1_list = []
            prop_0_sq_list = []
            prop_1_sq_list = []
            labels = growing_hypergraph.node_labels
            for edge in growing_hypergraph.edge_members:
                edge_labels = [labels[node] for node in edge]
                k_1_list.append(sum(edge_labels))
                k_0_list.append(len(edge_labels) - sum(edge_labels))
                k_list.append(len(edge_labels))
                prop_1_list.append(sum(edge_labels)/len(edge_labels))
                prop_0_list.append((len(edge_labels) - sum(edge_labels))/len(edge_labels))
                prop_1_sq_list.append((sum(edge_labels)**2)/len(edge_labels))
                prop_0_sq_list.append(((len(edge_labels) - sum(edge_labels))**2)/len(edge_labels))
            mean_k_0 =  sum(k_0_list)/len(k_0_list)
            mean_k_1 =  sum(k_1_list)/len(k_1_list)
            mean_k =  sum(k_list)/len(k_list)
            mean_prop_0_sq = sum(prop_0_sq_list)/len(prop_0_sq_list)
            mean_prop_1_sq = sum(prop_1_sq_list)/len(prop_1_sq_list)
            estimates.append([mean_prop_0_sq, ((mean_k_0 ** 2) + mean_k_0)/mean_k, mean_prop_1_sq, ((mean_k_1 ** 2) + mean_k_1)/mean_k])
        estimates = np.array(estimates)
        data.append([beta, sum(estimates[:,0]) / len(estimates[:,0]), sum(estimates[:,1]) / len(estimates[:,1]), 
                     sum(estimates[:,2]) / len(estimates[:,2]), sum(estimates[:,3]) / len(estimates[:,3])])
    return data

def generate_data_assm1(p, q, timesteps, experiments):
    beta_vals = np.arange(0, 1, step = 0.1)
    data = [["beta","actual k_0", "estimate k_0", "actual k_1", "estimate k_1", "actual k_0/k", "estimate k_0/k", "actual k_1/k", "estimate k_1/k", 
            "actual k_0^2/k", "estimate k_0^2/k", "actual k_1^2/k", "estimate k_1^2/k"]]
    for beta in beta_vals:
        print("beta: " + str(beta))
        estimates = []
        for experiment in range(experiments):
            H = xgi.Hypergraph([[0, 1]])
            H.set_node_attributes({0 : 0, 1 : 1}, name = "label")
            growing_hypergraph = GH(H, [0, 1], p, q)

            growing_hypergraph.add_hyperedge(timesteps, beta, 1 - beta, beta, 1 - beta)
            
            e_k_0_list = []
            e_k_1_list = []
            e_prop_0_list = []
            e_prop_1_list = []
            e_prop_0_sq_list = []
            e_prop_1_sq_list = []
            labels = growing_hypergraph.node_labels
            i_ests = []
            e_ests = []
            for edge_ind in range(1, len(growing_hypergraph.edge_members)):
                edge = growing_hypergraph.edge_members[edge_ind]
                i_k_0_list = []
                i_k_1_list = []
                i_prop_0_list = []
                i_prop_1_list = []
                i_prop_0_sq_list = []
                i_prop_1_sq_list = []
                prev_edges = growing_hypergraph.edge_members[0:edge_ind]
                for i in prev_edges:
                    i_labels = [labels[node] for node in i]
                    i_k_1_list.append(sum(i_labels))
                    i_k_0_list.append(len(i_labels) - sum(i_labels))
                    i_prop_1_list.append(sum(i_labels)/len(i_labels))
                    i_prop_0_list.append((len(i_labels) - sum(i_labels))/len(i_labels))
                    i_prop_1_sq_list.append((sum(i_labels)**2)/len(i_labels))
                    i_prop_0_sq_list.append(((len(i_labels) - sum(i_labels))**2)/len(i_labels))
                mean_k_0 =  sum(i_k_0_list)/len(i_k_0_list)
                mean_k_1 =  sum(i_k_1_list)/len(i_k_1_list)
                mean_prop_0 = sum(i_prop_0_list)/len(i_prop_0_list)
                mean_prop_1 = sum(i_prop_1_list)/len(i_prop_1_list)
                mean_prop_0_sq = sum(i_prop_0_sq_list)/len(i_prop_0_sq_list)
                mean_prop_1_sq = sum(i_prop_1_sq_list)/len(i_prop_1_sq_list)
                i_ests.append([mean_k_0, mean_k_1, mean_prop_0, mean_prop_1, mean_prop_0_sq, mean_prop_1_sq])

                e_labels = [labels[node] for node in edge]
                e_k_1_list.append(sum(e_labels))
                e_k_0_list.append(len(e_labels) - sum(e_labels))
                e_prop_1_list.append(sum(e_labels)/len(e_labels))
                e_prop_0_list.append((len(e_labels) - sum(e_labels))/len(e_labels))
                e_prop_1_sq_list.append((sum(e_labels)**2)/len(e_labels))
                e_prop_0_sq_list.append(((len(e_labels) - sum(e_labels))**2)/len(e_labels))
            mean_k_0 =  sum(e_k_0_list)/len(e_k_0_list)
            mean_k_1 =  sum(e_k_1_list)/len(e_k_1_list)
            mean_prop_0 = sum(e_prop_0_list)/len(e_prop_0_list)
            mean_prop_1 = sum(e_prop_1_list)/len(e_prop_1_list)
            mean_prop_0_sq = sum(e_prop_0_sq_list)/len(e_prop_0_sq_list)
            mean_prop_1_sq = sum(e_prop_1_sq_list)/len(e_prop_1_sq_list)
            i_ests = np.array(i_ests)
            e_ests.append([mean_k_0, sum(i_ests[:,0]) / len(i_ests[:,0]), mean_k_1, sum(i_ests[:,1]) / len(i_ests[:,1]), 
                           mean_prop_0, sum(i_ests[:,2]) / len(i_ests[:,2]), mean_prop_1, sum(i_ests[:,3]) / len(i_ests[:,3]), 
                           mean_prop_0_sq, sum(i_ests[:,4]) / len(i_ests[:,4]), mean_prop_1_sq, sum(i_ests[:,5]) / len(i_ests[:,5])])
        estimates = np.array(e_ests)
        data.append([beta, sum(estimates[:,0]) / len(estimates[:,0]), sum(estimates[:,1]) / len(estimates[:,1]), 
                sum(estimates[:,2]) / len(estimates[:,2]), sum(estimates[:,3]) / len(estimates[:,3]), 
                sum(estimates[:,4]) / len(estimates[:,4]), sum(estimates[:,5]) / len(estimates[:,5]),
                sum(estimates[:,6]) / len(estimates[:,6]), sum(estimates[:,7]) / len(estimates[:,7]),
                sum(estimates[:,8]) / len(estimates[:,8]), sum(estimates[:,9]) / len(estimates[:,9]),
                sum(estimates[:,10]) / len(estimates[:,10]), sum(estimates[:,11]) / len(estimates[:,11])])
    return data


data = generate_data_assm1(0.9, 0.1, 5000, 5)
with open('throughput/assumption1_09_01.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

data = generate_data_assm1(0.1, 0.9, 5000, 5)
with open('throughput/assumption1_01_09.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

data = generate_data_assm1(0.6, 0.4, 5000, 5)
with open('throughput/assumption1_06_04.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

data = generate_data_assm2(0.9, 0.1, 5000, 5)
with open('throughput/assumption2_09_01.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

data = generate_data_assm2(0.1, 0.9, 5000, 5)
with open('throughput/assumption2_01_09.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

data = generate_data_assm2(0.6, 0.4, 5000, 5)
with open('throughput/assumption2_06_04.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

data = generate_data_assm4(0.9, 0.1, 5000, 5)
with open('throughput/assumption4_09_01.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

data = generate_data_assm4(0.1, 0.9, 5000, 5)
with open('throughput/assumption4_01_09.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

data = generate_data_assm4(0.6, 0.4, 5000, 5)
with open('throughput/assumption4_06_04.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)