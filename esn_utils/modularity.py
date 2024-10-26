import numpy as np
import networkx as nx
import community as community_louvain

def total_weight(W):
    return np.sum(W)

def weighted_degree(W):
    return np.sum(W, axis=1)

def newman_modularity(W, community_assignments):
    num_nodes = W.shape[0]
    m = total_weight(W)
    k = weighted_degree(W)
    
    Q = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if community_assignments[i] == community_assignments[j]:
                Q += W[i, j] - (k[i] * k[j]) / (2 * m)
    
    return Q / (2 * m)

def create_graph_from_weight_matrix(W):
    G = nx.Graph()
    for i in range(W.shape[0]):
        for j in range(i, W.shape[1]):
            if W[i, j] > 0:
                G.add_edge(i, j, weight=W[i, j])
    return G

def find_community_assignments(G):
    partition = community_louvain.best_partition(G, weight='weight')
    num_nodes = len(G.nodes())
    community_assignments = [None] * num_nodes
    for node, community in partition.items():
        community_assignments[node] = community
    return community_assignments

def compute_modularity(W):
    W = np.where(W != 0, 1, 0)
    G = create_graph_from_weight_matrix(W)
    if len(G.nodes()) < W.shape[0]:
        # print("Graph is not connected")
        return 0
    else:
        community_assignments = find_community_assignments(G)
        return newman_modularity(W, community_assignments)