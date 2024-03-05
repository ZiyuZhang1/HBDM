import scanpy as sc
import pandas as pd
import numpy as np
import pickle
import networkx as nx
import os
import scipy.sparse as sp
import matplotlib.pyplot as plt
 
dataset = 'sc_vsmc_health_hv'

with open(r'D:/study/thesis/project/HBDM-main/data/'+dataset+'/sc_sub.p', 'rb') as f:
    sc_sub = pickle.load(f)
gene_pair = sc_sub.adjacency
gene_pair = gene_pair.round(2)
st_idx = gene_pair.columns.to_numpy()
st_arr = gene_pair.to_numpy()
st_arr[st_arr < 0.01] = 0

##### create hbdm inputs
root = 'D:/study/thesis/project/HBDM-main/data/datasets/'+dataset+'/'
if not os.path.exists(root):
    os.makedirs(root)
else:
    pass
 
 
# create sparse matrix from st_arr_new
sparse_matrix = sp.csr_matrix(st_arr)
 
# get upper triangle as a sparse array
sparse_upper = sp.triu(sparse_matrix, k=1)
i, j, data = sp.find(sparse_upper)
 
col1 = st_idx[i]
col2 = st_idx[j]
graph_df = pd.DataFrame({'combined_score': data})
graph_df['protein1'] = col1
graph_df['protein2'] = col2
graph_df = graph_df[graph_df['combined_score'] != 0]
graph_df['combined_score'] = graph_df['combined_score'].round(4)

G = nx.from_pandas_edgelist(graph_df, source='protein1', target='protein2', edge_attr='combined_score', create_using=nx.Graph)

H = nx.minimum_spanning_tree(G)
existing_edges = list(G.edges())
sample_pool = list(set(existing_edges)-set(H.edges()))
edge_weights = [G[edge[0]][edge[1]]['combined_score'] for edge in sample_pool]
df = pd.DataFrame({'Node Pair': sample_pool, 'Edge Weight': edge_weights})


chosen_paris = df[df['Edge Weight']>=0.04]['Node Pair']

remove_pairs = set(sample_pool) - set(chosen_paris.tolist())
G.remove_edges_from(remove_pairs)
graph_df = nx.to_pandas_edgelist(G, source='protein1', target='protein2')
proteins = sorted(list(set(graph_df['protein1'].tolist())|set(graph_df['protein2'].tolist())))
gene2node = {value: index for index, value in enumerate(proteins)}
 
file_path = root+'ppi_index.pkl'
# Serialize and save the Tensor to the file
with open(file_path, 'wb') as file:
    pickle.dump(gene2node, file)
# Close the file
file.close()
 
graph_df['node1']=graph_df['protein1'].map(gene2node)
graph_df['node2']=graph_df['protein2'].map(gene2node)
 
G = nx.from_pandas_edgelist(graph_df, source='node1', target='node2', edge_attr='combined_score', create_using=nx.Graph)
 
print(len(G.edges),len(G.nodes))
 
 
def plot_degree_distribution(name, G, use_weight=False):
    """
    Plot the degree distribution of a graph.
 
    Parameters:
    - G: NetworkX graph
    - use_weight: Boolean, whether to consider edge weights in the degree calculation
    """
 
    # Compute the degree distribution
    if use_weight:
        print('weight')
        degree_sequence = sorted([d for n, d in G.degree(weight="weight")], reverse=True)
    else:
        degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
 
    degree_counts = nx.degree_histogram(G)
 
    # Convert counts to fractions
    num_nodes = len(G.nodes)
    degree_fraction = [count / num_nodes for count in degree_counts]
 
    # Plot the degree distribution on a log-log scale
    plt.loglog(range(len(degree_fraction)), degree_fraction, 'o', markersize=5)
    plt.title(name+" Degree Distribution")
    plt.xlabel("Degree (log scale)")
    plt.ylabel("Fraction of Nodes (log scale)")
    plt.savefig(root+'degree.png')
    plt.show()
plot_degree_distribution(dataset,G, use_weight=False)
 
degree_sequence = [d for n, d in G.degree()]
# Calculate the average degree
avg_degree = sum(degree_sequence) / len(degree_sequence)
print(avg_degree)
 
edges = np.array([(u, v, data['combined_score']) for u, v, data in G.edges(data=True)])
 
# Determine i, j, and weights
i = np.where(edges[:, 0] > edges[:, 1], edges[:, 1], edges[:, 0])
j = np.where(edges[:, 0] > edges[:, 1], edges[:, 0], edges[:, 1])
weights = edges[:, 2]
 
 
np.savetxt(root+'sparse_i.txt', np.array(i), delimiter='\n')
np.savetxt(root+'sparse_j.txt', np.array(j), delimiter='\n')
np.savetxt(root+'sparse_w.txt', np.array(weights), delimiter='\n')
 
level_edges = dict()
for u, v, data in G.edges(data=True):
    level = int(str(data['combined_score']*10)[0])
    if level in level_edges:
        level_edges[level].append([u, v])
    else:
        level_edges[level]=[[u, v]]
 
for level in level_edges:
    edges = np.array(level_edges[level])
    sparse_i = np.where(edges[:, 0] > edges[:, 1], edges[:, 1], edges[:, 0])
    sparse_j = np.where(edges[:, 0] > edges[:, 1], edges[:, 0], edges[:, 1])
    np.savetxt(root+'level_'+str(level)+'_sparse_i.txt', np.array(sparse_i), delimiter='\n')
    np.savetxt(root+'level_'+str(level)+'_sparse_j.txt', np.array(sparse_j), delimiter='\n')