import xgi
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import math
import scipy.special as ss; ss.binom

class GH:
    def __init__(self, H, labels, p, q):
        self.H = H
        self.labels = labels
        self.p = p
        self.q = q
        self.edge_members = H.edges.members()
        self.node_labels = H.nodes.attrs("label").aslist()
        self.nodes = list(H.nodes)
        self.last_added = [max(H.nodes)] * len(H.edges)
        self.first_seen = self.initialize_first_seen()
        self.total_num_1 = sum(self.node_labels)
        self.total_num_0 = len(self.node_labels) - self.total_num_1

    def get_labels(self):
        return(self.node_labels)
    
    def get_edges(self):
        return(self.edge_members)
    
    def initialize_first_seen(self):
        # stores the edge index where each node first appears
        first_seen = [len(self.get_edges())] * len(self.get_labels())

        for e_index in range(len(self.get_edges())):
            for node in self.get_edges()[e_index]:
                first_seen[node] = min(first_seen[node], e_index)

        return first_seen

    
    def set_values(self, value0, value1, u_label):
        if u_label == 0:
            return([value0, value1])
        else: 
            return([value1, value0])
        
    def return_key_values(self, e_index, u_index, e_prime_index, theta):
        p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er = theta

        node_labels = self.node_labels

        e = self.edge_members[e_index]
        e_prime = self.edge_members[e_prime_index]
        u_label = node_labels[u_index]

        intersect = e.intersection(e_prime)

        if len(intersect) == 0:
            return(0)
        if u_index not in e_prime:
            return(0)

        node_labels = self.get_labels()
        e_labels = [node_labels[node] for node in e]
        e_prime_labels = [node_labels[node] for node in e_prime]
        int_labels = [node_labels[node] for node in intersect]

        # Get edge and intersection sizes
        e_prime_num_1 = sum(e_prime_labels)
        e_prime_num_0 = len(e_prime_labels) - e_prime_num_1
        e_num_1 = sum(e_labels)
        e_num_0 = len(e_labels) - e_num_1
        e_num_u, e_num_r = self.set_values(e_num_0, e_num_1, u_label)
        e_prime_num_u, e_prime_num_r = self.set_values(e_prime_num_0, e_prime_num_1, u_label)
        int_num_u, int_num_r = self.set_values(len(int_labels) - sum(int_labels), sum(int_labels), u_label)

        # Get the new nodes
        prev_edges = self.edge_members[0:e_prime_index]
        prev_nodes = list(range(self.last_added[e_prime_index - 1] + 1))

        novel_nodes = set(e_prime) - set(prev_nodes)
        novel_labels = [node_labels[node] for node in novel_nodes]

        novel_num_u, novel_num_r = self.set_values(len(novel_labels) - sum(novel_labels), sum(novel_labels), u_label)
      
        # Get the external nodes
        # total_num_1 = sum(node_labels)
        # total_num_0 = len(node_labels) - total_num_1
        all_ext_num_0 = self.total_num_0 - e_num_0
        all_ext_num_1 = self.total_num_1 - e_num_1

        # external_nodes = set(prev_nodes).intersection(e_prime) - set(e)
        # external_labels = [node_labels[node] for node in external_nodes]           
        ext_num_u = e_prime_num_u - int_num_u - novel_num_u
        ext_num_r = e_prime_num_r - int_num_r - novel_num_r


        all_ext_num_u, all_ext_num_r = self.set_values(all_ext_num_0, all_ext_num_1, u_label)
        # ext_num_u, ext_num_r = self.set_values(len(external_labels) - sum(external_labels), sum(external_labels), u_label)

        # Probability calculation
        prob_e = 1 / e_prime_index

        prob_u = 1 / len(e)

        P1 = p ** (int_num_u - 1)
        P2 = (1 - p) ** (e_num_u - int_num_u)
        P3 = q ** (int_num_r)
        P4 = (1 - q) ** (e_num_r - int_num_r)

        if ext_num_u == 0:
            prob_those_ext_u = 1
        else:
            prob_those_ext_u = 1 / ss.binom(all_ext_num_u, ext_num_u) # formerly: ext_num_u / all_ext_num_u
        if ext_num_r == 0:
            prob_those_ext_r = 1
        else: prob_those_ext_r = 1 / ss.binom(all_ext_num_r, ext_num_r) # formerly: ext_num_r / all_ext_num_r

        P5_numer = (math.e ** (-gamma_eu)) * (gamma_eu ** ext_num_u) * prob_those_ext_u * (math.e ** (-gamma_er)) * (gamma_er ** ext_num_r) * prob_those_ext_r
        P5_denom = math.factorial(ext_num_u) * math.factorial(ext_num_r)
        P5 = P5_numer / P5_denom
        P6_numer = (math.e ** (-gamma_nu)) * (gamma_nu ** novel_num_u) * (math.e ** (-gamma_nr)) * (gamma_nr ** novel_num_r)
        P6_denom = (math.factorial(novel_num_u) * math.factorial(novel_num_r))
        P6 = P6_numer / P6_denom
        prob_e_prime = P1 * P2 * P3 * P4  * P5 * P6
        
        prob = prob_e * prob_u * prob_e_prime

        return(prob)        
        
    def likelihood(self, e_index, u_index, e_prime_index, theta):
        p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er = theta

        node_labels = self.node_labels

        e = self.edge_members[e_index]
        e_prime = self.edge_members[e_prime_index]
        u_label = node_labels[u_index]

        intersect = e.intersection(e_prime)

        if len(intersect) == 0:
            return(0)
        if u_index not in e_prime:
            return(0)

        node_labels = self.get_labels()
        e_labels = [node_labels[node] for node in e]
        e_prime_labels = [node_labels[node] for node in e_prime]
        int_labels = [node_labels[node] for node in intersect]

        # Get edge and intersection sizes
        e_prime_num_1 = sum(e_prime_labels)
        e_prime_num_0 = len(e_prime_labels) - e_prime_num_1
        e_num_1 = sum(e_labels)
        e_num_0 = len(e_labels) - e_num_1
        e_num_u, e_num_r = self.set_values(e_num_0, e_num_1, u_label)
        e_prime_num_u, e_prime_num_r = self.set_values(e_prime_num_0, e_prime_num_1, u_label)
        int_num_u, int_num_r = self.set_values(len(int_labels) - sum(int_labels), sum(int_labels), u_label)

        # Get the new nodes
        prev_edges = self.edge_members[0:e_prime_index]
        prev_nodes = list(range(self.last_added[e_prime_index - 1] + 1))

        novel_nodes = set(e_prime) - set(prev_nodes)
        novel_labels = [node_labels[node] for node in novel_nodes]

        novel_num_u, novel_num_r = self.set_values(len(novel_labels) - sum(novel_labels), sum(novel_labels), u_label)
      
        # Get the external nodes
        # total_num_1 = sum(node_labels)
        # total_num_0 = len(node_labels) - total_num_1

        # PREVIOUS
        # all_ext_num_0 = self.total_num_0 - e_num_0
        # all_ext_num_1 = self.total_num_1 - e_num_1

        # NEW REVISIONS
        all_ext_nodes = set(prev_nodes) - e
        all_ext_labels = [node_labels[node] for node in all_ext_nodes]

        all_ext_num_1 = sum(all_ext_labels)
        all_ext_num_0 = len(all_ext_labels) - sum(all_ext_labels)


        # external_nodes = set(prev_nodes).intersection(e_prime) - set(e)
        # external_labels = [node_labels[node] for node in external_nodes]           
        ext_num_u = e_prime_num_u - int_num_u - novel_num_u
        ext_num_r = e_prime_num_r - int_num_r - novel_num_r


        all_ext_num_u, all_ext_num_r = self.set_values(all_ext_num_0, all_ext_num_1, u_label)
        # ext_num_u, ext_num_r = self.set_values(len(external_labels) - sum(external_labels), sum(external_labels), u_label)

        # Probability calculation
        prob_e = 1 / e_prime_index

        prob_u = 1 / len(e)

        P1 = p ** (int_num_u - 1)
        P2 = (1 - p) ** (e_num_u - int_num_u)
        P3 = q ** (int_num_r)
        P4 = (1 - q) ** (e_num_r - int_num_r)

        if ext_num_u == 0:
            prob_those_ext_u = 1
        else:
            prob_those_ext_u = 1 / ss.binom(all_ext_num_u, ext_num_u) # formerly: ext_num_u / all_ext_num_u
        if ext_num_r == 0:
            prob_those_ext_r = 1
        else: prob_those_ext_r = 1 / ss.binom(all_ext_num_r, ext_num_r) # formerly: ext_num_r / all_ext_num_r

        P5_numer = (math.e ** (-gamma_eu)) * (gamma_eu ** ext_num_u) * prob_those_ext_u * (math.e ** (-gamma_er)) * (gamma_er ** ext_num_r) * prob_those_ext_r
        P5_denom = math.factorial(ext_num_u) * math.factorial(ext_num_r)
        P5 = P5_numer / P5_denom
        P6_numer = (math.e ** (-gamma_nu)) * (gamma_nu ** novel_num_u) * (math.e ** (-gamma_nr)) * (gamma_nr ** novel_num_r)
        P6_denom = (math.factorial(novel_num_u) * math.factorial(novel_num_r))
        P6 = P6_numer / P6_denom
        prob_e_prime = P1 * P2 * P3 * P4  * P5 * P6
        
        prob = prob_e * prob_u * prob_e_prime

        return(prob)
    
    def f_likelihood(self, e_index, u_index, e_prime_index, theta):
        p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er = theta

        node_labels = self.node_labels

        e = self.edge_members[e_index]
        e_prime = self.edge_members[e_prime_index]
        u_label = node_labels[u_index]

        intersect = e.intersection(e_prime)

        if len(intersect) == 0:
            return(0)
        if u_index not in e_prime:
            return(0)

        node_labels = self.get_labels()
        e_labels = [node_labels[node] for node in e]
        e_prime_labels = [node_labels[node] for node in e_prime]
        int_labels = [node_labels[node] for node in intersect]

        # Get edge and intersection sizes
        e_prime_num_1 = sum(e_prime_labels)
        e_prime_num_0 = len(e_prime_labels) - e_prime_num_1
        e_num_1 = sum(e_labels)
        e_num_0 = len(e_labels) - e_num_1
        e_num_u, e_num_r = self.set_values(e_num_0, e_num_1, u_label)
        e_prime_num_u, e_prime_num_r = self.set_values(e_prime_num_0, e_prime_num_1, u_label)
        int_num_u, int_num_r = self.set_values(len(int_labels) - sum(int_labels), sum(int_labels), u_label)

        # Get the new nodes
        prev_edges = self.edge_members[0:e_prime_index]
        prev_nodes = list(range(self.last_added[e_prime_index - 1] + 1))

        novel_nodes = set(e_prime) - set(prev_nodes)
        novel_labels = [node_labels[node] for node in novel_nodes]

        novel_num_u, novel_num_r = self.set_values(len(novel_labels) - sum(novel_labels), sum(novel_labels), u_label)
      
        # Get the external nodes
        # total_num_1 = sum(node_labels)
        # total_num_0 = len(node_labels) - total_num_1
        all_ext_num_0 = self.total_num_0 - e_num_0
        all_ext_num_1 = self.total_num_1 - e_num_1

        # external_nodes = set(prev_nodes).intersection(e_prime) - set(e)
        # external_labels = [node_labels[node] for node in external_nodes]           
        ext_num_u = e_prime_num_u - int_num_u - novel_num_u
        ext_num_r = e_prime_num_r - int_num_r - novel_num_r


        all_ext_num_u, all_ext_num_r = self.set_values(all_ext_num_0, all_ext_num_1, u_label)
        # ext_num_u, ext_num_r = self.set_values(len(external_labels) - sum(external_labels), sum(external_labels), u_label)

        # Probability calculation
        prob_e = 1 / e_prime_index

        prob_u = 1 / len(e)

        P1 = p ** (int_num_u - 1)
        P2 = (1 - p) ** (e_num_u - int_num_u)
        P3 = q ** (int_num_r)
        P4 = (1 - q) ** (e_num_r - int_num_r)

        if ext_num_u == 0:
            prob_those_ext_u = 1
        else:
            prob_those_ext_u = 1 / ss.binom(all_ext_num_u, ext_num_u) # formerly: ext_num_u / all_ext_num_u
        if ext_num_r == 0:
            prob_those_ext_r = 1
        else: prob_those_ext_r = 1 / ss.binom(all_ext_num_r, ext_num_r) # formerly: ext_num_r / all_ext_num_r

        P5_numer = (math.e ** (-gamma_eu)) * (gamma_eu ** ext_num_u) * prob_those_ext_u * (math.e ** (-gamma_er)) * (gamma_er ** ext_num_r) * prob_those_ext_r
        P5_denom = math.factorial(ext_num_u) * math.factorial(ext_num_r)
        P5 = P5_numer / P5_denom
        P6_numer = (math.e ** (-gamma_nu)) * (gamma_nu ** novel_num_u) * (math.e ** (-gamma_nr)) * (gamma_nr ** novel_num_r)
        P6_denom = (math.factorial(novel_num_u) * math.factorial(novel_num_r))
        P6 = P6_numer / P6_denom
        prob_e_prime = P1 * P2 * P3 * P4  * P5 * P6
        
        lik = prob_e * prob_u * prob_e_prime

        # get s_u and t_u
        s_u = e_num_u - 1
        t_u = int_num_u - 1

        # get s_r and t_r
        s_r = e_num_r
        t_r = int_num_r

        f = np.array([t_u, s_u, t_r, s_r, ext_num_u, ext_num_r, novel_num_u, novel_num_r])

        return([lik, f])

    def add_hyperedge(self, num_edges = 1, gamma_nu = 1, gamma_nr = 1, gamma_eu = 1, gamma_er = 1):
        for i in range(num_edges):
            e_prime = [] # create an empty new edge called e_prime

            ## randomly select an existing hyperedge e to start with
            e_num = random.randint(0, len(self.edge_members) - 1)
            e = self.edge_members[e_num]
            e_size = len(e)

            ## randomly select a node u from e
            u_num = random.randint(0, e_size - 1)
            u = list(e)[u_num]
            u_label = self.node_labels[u]
            e_prime.append(u)

            ## get r label
            if self.labels[0] == u_label:
                r_label = self.labels[1]
            else:
                r_label = self.labels[0]

            ## add other nodes from e to e_prime
            # add node with same label as u with prob p
            # add node with different label from u with prob q
            for node in e:
                if self.node_labels[node] == u_label:
                    prob = self.p
                else:
                    prob = self.q
                if random.random() < prob:
                    e_prime.append(node)
            
            ## add exterior nodes
            num_ext_u = np.random.poisson(gamma_eu, 1)[0]
            num_ext_r = np.random.poisson(gamma_er, 1)[0]

            u_indices = list(filter(lambda x: self.node_labels[x] == u_label, self.nodes))
            r_indices = list(filter(lambda x: self.node_labels[x] != u_label, self.nodes))

            for i in range(0, num_ext_u):
                if len(set(u_indices) - set(e)) > 0:
                    exterior_node = random.sample(list(set(u_indices) - set(e)), 1)[0] # randomly sample a node from outside the existing hyperedge
                    e_prime.append(exterior_node)

            for i in range(0, num_ext_r):
                if len(set(r_indices) - set(e)) > 0:
                    exterior_node = random.sample(list(set(r_indices) - set(e)), 1)[0] # randomly sample a node from outside the existing hyperedge
                    e_prime.append(exterior_node)

            ## Add new nodes        
            num_new_u = np.random.poisson(gamma_nu, 1)[0]
            num_new_r = np.random.poisson(gamma_nr, 1)[0]

            last = self.last_added[-1]
            for i in range(0, num_new_u):
                new_node = len(self.nodes)
                self.nodes.append(new_node)
                self.node_labels.append(u_label)
                e_prime.append(new_node)
                last = new_node
                if u_label == 0:
                    self.total_num_0 += 1
                else:
                    self.total_num_1 += 1

            for i in range(0, num_new_r):
                new_node = len(self.nodes)
                self.nodes.append(new_node)
                self.node_labels.append(r_label)
                e_prime.append(new_node)
                last = new_node
                if u_label == 1:
                    self.total_num_0 += 1
                else:
                    self.total_num_1 += 1

            ## Add the edge to the hypergraph
            self.H.add_edge(e_prime)
            self.edge_members.append(set(e_prime))
            self.last_added.append(last)

            self.first_seen = self.initialize_first_seen()

        # big_H = xgi.Hypergraph(self.edge_members)
        # node_dict = dict(zip(self.nodes, self.node_labels))
        # big_H.set_node_attributes(node_dict, name = "label")
        # return(big_H)
        
    def generate_canidate_f(self, e_index):
        canidate_f = []
        e = self.get_edges()[e_index]
        for f_index in range(e_index):
            if len(self.get_edges()[f_index].intersection(e)) != 0:
                canidate_f.append(f_index)
        
        return canidate_f
    
    def p_e_given_f(self, theta, labels, e_index, f_index):
        e = self.get_edges()[e_index]
        f = self.get_edges()[f_index]

        node_labels = labels
        prev_nodes = list(range(self.last_added[e_index - 1] + 1))
        p, q, gamma_nu, gamma_nr, gamma_eu, gamma_er = theta

        # novel nodes counting
        novel_nodes = set(e) - set(prev_nodes)
        novel_labels = [node_labels[node] for node in novel_nodes]
        nov1 = sum(novel_labels)
        nov0 = len(novel_labels) - nov1

        # external nodes counting
        external_nodes = set(prev_nodes) - set(f)
        external_node_labels = [node_labels[node] for node in external_nodes]  

        external_nodes_added = external_nodes.intersection(e)
        external_nodes_added_labels = [node_labels[node] for node in external_nodes_added] 

        ext1 = sum(external_nodes_added_labels)
        ext0 = len(external_nodes_added_labels) - ext1

        posext1 = sum(external_node_labels)
        posext0 = len(external_node_labels) - posext1

        # copied and not copied nodes counting
        copied_nodes = e.intersection(f)
        copied_nodes_labels = [node_labels[node] for node in copied_nodes]

        cop1 = sum(copied_nodes_labels)
        cop0 = len(copied_nodes_labels) - cop1

        not_copied_nodes = set(f) - set(e)
        not_copied_nodes_labels = [node_labels[node] for node in not_copied_nodes]

        notcop1 = sum(not_copied_nodes_labels)
        notcop0 = len(not_copied_nodes_labels) - notcop1

        # find probs of u being label 1 and 0
        possible_u_labels = [node_labels[node] for node in f.intersection(e)]
        prob_u_label_equals_1 = sum(possible_u_labels) / len(possible_u_labels)
        prob_u_label_equals_0 = 1 - prob_u_label_equals_1

        # calculate prob of e given f and u_label = 1
        prob_given_f_u_label_1 = (p**(cop1-1)) * ((1-p)**notcop1) * (q**cop0) * ((1-q)**notcop0)
        prob_given_f_u_label_1 *= (gamma_eu**ext1) * math.exp(-gamma_eu) / math.factorial(ext1) / math.comb(posext1, ext1)
        prob_given_f_u_label_1 *= (gamma_er**ext0) * math.exp(-gamma_er) / math.factorial(ext0) / math.comb(posext0, ext0)
        prob_given_f_u_label_1 *= (gamma_nu**nov1) * math.exp(-gamma_nu) / math.factorial(nov1)
        prob_given_f_u_label_1 *= (gamma_nr**nov0) * math.exp(-gamma_nr) / math.factorial(nov0)

        # calculate prob of e given f and u_label = 0
        prob_given_f_u_label_0 = (p**(cop0-1)) * ((1-p)**notcop0) * (q**cop1) * ((1-q)**notcop1)
        prob_given_f_u_label_0 *= (gamma_eu**ext0) * math.exp(-gamma_eu) / math.factorial(ext0) / math.comb(posext0, ext0)
        prob_given_f_u_label_0 *= (gamma_er**ext1) * math.exp(-gamma_er) / math.factorial(ext1) / math.comb(posext1, ext1)
        prob_given_f_u_label_0 *= (gamma_nu**nov0) * math.exp(-gamma_nu) / math.factorial(nov0)
        prob_given_f_u_label_0 *= (gamma_nr**nov1) * math.exp(-gamma_nr) / math.factorial(nov1)

        # put together, return full expression
        return prob_u_label_equals_1*prob_given_f_u_label_1 + prob_u_label_equals_0*prob_given_f_u_label_0
    
    def total_log_likelihood_complete(self, theta, labels):
        # e_index 0 must be given...

        summation = 0
        for e_index in range(1, len(self.get_edges())):
            # add all edges together
            e_sum = 0
            canidate_f_indexes = self.generate_canidate_f(e_index)
            for f_index in canidate_f_indexes:
                e_sum += self.p_e_given_f(theta, labels, e_index, f_index)
            
            e_sum /= e_index # divide by possible f_choices
            summation += np.log(e_sum)

        return summation