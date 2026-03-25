import numpy as np
import random
from src.poisson_hypergraph import GH
import xgi
import matplotlib.pyplot as plt 
import math


class sem_functions:

    def f(self, GH, e_index, u_index, e_prime_index):
        labels = GH.get_labels()

        e = GH.edge_members[e_index]
        e_prime = GH.edge_members[e_prime_index]
        u_label = labels[u_index]

        intersect = e.intersection(e_prime)

        e_labels = [labels[node] for node in e]
        e_prime_labels = [labels[node] for node in e_prime]
        int_labels = [labels[node] for node in intersect]

        int_num_u, int_num_r = GH.set_values(len(int_labels) - sum(int_labels), sum(int_labels), u_label)

        e_num_1 = sum(e_labels)
        e_num_0 = len(e_labels) - e_num_1
        e_num_u, e_num_r = GH.set_values(e_num_0, e_num_1, u_label)

        # get s_u and t_u
        s_u = e_num_u - 1
        t_u = int_num_u - 1

        # get s_r and t_r
        s_r = e_num_r
        t_r = int_num_r

        # Get the novel nodes
        prev_nodes = list(range(GH.last_added[e_prime_index - 1] + 1))
        novel_nodes = set(e_prime) - set(prev_nodes)
        novel_labels = [labels[node] for node in novel_nodes]

        novel_num_u, novel_num_r = GH.set_values(len(novel_labels) - sum(novel_labels), sum(novel_labels), u_label)

        # Get the external nodes
        e_prime_num_1 = sum(e_prime_labels)
        e_prime_num_0 = len(e_prime_labels) - e_prime_num_1
        e_prime_num_u, e_prime_num_r = GH.set_values(e_prime_num_0, e_prime_num_1, u_label)

        ext_num_u = e_prime_num_u - int_num_u - novel_num_u
        ext_num_r = e_prime_num_r - int_num_r - novel_num_r

        return(np.array([t_u, s_u, t_r, s_r, ext_num_u, ext_num_r, novel_num_u, novel_num_r]))


    # exp_stats computes the expected sufficient statistics given an e' and some theta vals
    def exp_stats(self, GH, e_prime_index, theta):
        edge_members = GH.get_edges()
        e_prime = edge_members[e_prime_index]

        suff_stats_num = np.zeros(8)
        suff_stats_denom = np.zeros(8)

        for e_index in range(e_prime_index):
            e = edge_members[e_index]
            for u_index in set(e_prime).intersection(set(e)):
                lik_f = GH.f_likelihood(e_index, u_index, e_prime_index, theta)
                suff_stats_num += lik_f[1] * lik_f[0]
                suff_stats_denom += lik_f[0]
                # lik = GH.likelihood(e_index, u_index, e_prime_index, theta)
                # suff_stats_num += self.f(GH, e_index, u_index, e_prime_index) * lik
                # suff_stats_denom += lik
        
        if 0 in suff_stats_denom:
            return suff_stats_denom

        exp_suff_stats = suff_stats_num / suff_stats_denom
        return exp_suff_stats

    def g(self, s):
        t_u, s_u, t_r, s_r, ext_num_u, ext_num_r, novel_num_u, novel_num_r = s
        if s_u == 0: 
            p = 1
        else:
            p = t_u / s_u
        if s_r == 0:
            q = 1
        else:
            q = t_r / s_r
        gamma_nu = novel_num_u
        gamma_nr = novel_num_r
        gamma_eu = ext_num_u
        gamma_er = ext_num_r
        return np.array([p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er])
    
    def e_prime_prob(self, GH, e_prime_index, theta):
        edges = GH.get_edges()
        e_prime = edges[e_prime_index]
        summation = 0
        for e_index in range(e_prime_index):
            e = edges[e_index]
            for u_index in e.intersection(e_prime):
                summation += GH.likelihood(e_index, u_index, e_prime_index, theta)
        return summation
    
    def GH_prob(self, GH, theta):
        edges = GH.get_edges()
        this_likelihood = 0
        for e_prime_index in range(1, len(edges)):
            this_likelihood += np.log(self.e_prime_prob(GH, e_prime_index, theta))
        return this_likelihood

    def SEM_with_likelihood(self, GH, s, timesteps, initial_rate, constant):
        p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er = self.g(s)
        edges = GH.get_edges()
        lr = initial_rate
        estimates = [[0, p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er]]
        likelihoods = [self.GH_prob(GH, [p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er])]
        for t in range(1, timesteps):
            e_prime_index = random.randint(1, len(edges) - 1)
            s_prime = self.exp_stats(GH, e_prime_index, [p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er])
            lr = lr * (math.e ** (-constant))
            s = ((1 - lr) * s) + (lr * s_prime)
            p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er  = self.g(s)
            estimates.append([t, p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er])
            likelihoods.append(self.GH_prob(GH, [p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er]))
            if t > 400:
                i = 2
                while False not in (np.abs(np.array(estimates[-1][1:]) - np.array(estimates[-i][1:])) < 0.05):
                    if i == 400:
                        return estimates, likelihoods
                    i += 1
        return estimates, likelihoods
    
    def SEM_without_likelihood(self, GH, s, timesteps, initial_rate, constant):
        p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er = self.g(s)
        edges = GH.get_edges()
        lr = initial_rate
        estimates = [[0, p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er]]
        # likelihoods = [self.GH_prob(GH, [p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er])]
        for t in range(1, timesteps):
            if t % 100 == 0:
                print(t)
            e_prime_index = random.randint(1, len(edges) - 1)
            s_prime = self.exp_stats(GH, e_prime_index, [p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er])
            lr = lr * (math.e ** (-constant))
            s = ((1 - lr) * s) + (lr * s_prime)
            p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er  = self.g(s)
            estimates.append([t, p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er])
            # likelihoods.append(self.GH_prob(GH, [p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er]))
            if t > 400:
                all_converged = True
                for i in range(2, 402):
                    if not np.allclose(
                        estimates[-1][1:],
                        estimates[-i][1:],
                        rtol=0.5,
                        atol=1e-2     # absolute floor for near-zero params
                    ):
                        all_converged = False
                        break
                if all_converged:
                    print("criterion met! iteration " + str(t))
                    return estimates
        return estimates #, likelihoods

    def viz(self, true_values, estimates, title):
        fig, axs = plt.subplots(1, 3)
        fig.set_figwidth(20)
        fig.set_figheight(7)
        fig.suptitle(title)
        axs[0].set_title('p and q')
        axs[0].plot(estimates[:, 0], [true_values[0]] * len(estimates[:, 0]), c = 'black', linestyle = 'dotted', label = "true values")
        axs[0].plot(estimates[:, 0], [true_values[1]] * len(estimates[:, 0]), c = 'black', linestyle = 'dotted')
        axs[0].plot(estimates[:, 0], estimates[:, 1], label = "estimate of param 1")
        axs[0].plot(estimates[:, 0], estimates[:, 2], label = "estimate of param 2")
        axs[0].set_xlabel("timesteps")
        axs[0].set_ylabel("param value")
        axs[1].set_title('$\\gamma_{NU}$ and $\\gamma_{NR}$')
        axs[1].plot(estimates[:, 0], [true_values[2]] * len(estimates[:, 0]), c = 'black', linestyle = 'dotted')
        axs[1].plot(estimates[:, 0], [true_values[3]] * len(estimates[:, 0]), c = 'black', linestyle = 'dotted')
        axs[1].plot(estimates[:, 0], estimates[:, 3])
        axs[1].plot(estimates[:, 0], estimates[:, 4])
        axs[1].set_xlabel("timesteps")
        axs[1].set_ylabel("param value")
        axs[2].set_title('$\\gamma_{EU}$ and $\\gamma_{ER}$')
        axs[2].plot(estimates[:, 0], [true_values[4]] * len(estimates[:, 0]), c = 'black', linestyle = 'dotted')
        axs[2].plot(estimates[:, 0], [true_values[5]] * len(estimates[:, 0]), c = 'black', linestyle = 'dotted')
        axs[2].plot(estimates[:, 0], estimates[:, 5])
        axs[2].plot(estimates[:, 0], estimates[:, 6])
        axs[2].set_xlabel("timesteps")
        axs[2].set_ylabel("param value")
        fig.legend()
        return plt

    def generate_hypergraph(self, theta, size):
        H = xgi.Hypergraph([[0, 1]])
        H.set_node_attributes({0 : 0, 1 : 1}, name = "label")
        growing_hypergraph = GH(H, [0, 1], theta[0], theta[1])
        growing_hypergraph.add_hyperedge(size, theta[2], theta[3], theta[4], theta[5])
        return growing_hypergraph

    def generate_hypergraph_big(self, theta, size):
        H = xgi.Hypergraph([[0, 1], [0, 2, 3, 4], [1, 5, 6, 7, 8]])
        H.set_node_attributes({0 : 0, 1 : 1, 2 : 0, 3 : 1, 4 : 0, 5 : 1, 6 : 0, 7 : 1, 8 : 0}, name = "label")
        growing_hypergraph = GH(H, [0, 1], theta[0], theta[1])
        growing_hypergraph.add_hyperedge(size, theta[2], theta[3], theta[4], theta[5])
        return growing_hypergraph

