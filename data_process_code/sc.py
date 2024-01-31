#!/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/hbdm_env/bin/python3.8
#SBATCH -J sc_process
#SBATCH -o sc_process.log
#SBATCH -e sc_process.err
#SBATCH --partition=compute
#SBATCH --time=5:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=80G

import numpy as np
import pandas as pd
import pickle
import networkx as nx
import os
import matplotlib.pylab as plt

dataset = 'sc'

root = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/HBDM/data/datasets/'+dataset+'/'
if not os.path.exists(root):
    os.makedirs(root)
else:
    pass

# with open('/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/co_exp/T3_sc_all_cell_coxp_matrix.pkl', 'rb') as f:
#     # load data from pickle file
#     st = pickle.load(f)
# st_arr = st[0]
# st_idx= st[1]

# import scipy.sparse as sp

# # create sparse matrix from st_arr_new
# sparse_matrix = sp.csr_matrix(st_arr)

# # get upper triangle as a sparse array
# sparse_upper = sp.triu(sparse_matrix, k=1)
# i, j, data = sp.find(sparse_upper)

# col1 = st_idx[i]
# col2 = st_idx[j]
# graph_df = pd.DataFrame({'combined_score': data})
# graph_df['protein1'] = col1.values
# graph_df['protein2'] = col2.values
# graph_df['combined_score'] = graph_df['combined_score'].round(3)


graph_df = pd.read_csv('/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/data/sc_all.csv')
graph_df  = graph_df.rename(columns={'coexp_value': 'combined_score', 'gene1': 'protein1', 'gene2': 'protein2'})
proteins = sorted(list(set(graph_df['protein1'].tolist())|set(graph_df['protein2'].tolist())))
gene2node = {value: index for index, value in enumerate(proteins)}


# file_path = root+'ppi_index.pkl'
# # Serialize and save the Tensor to the file
# with open(file_path, 'wb') as file:
#     pickle.dump(gene2node, file)
# # Close the file
# file.close()

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

# edges = np.array([(u, v, data['combined_score']) for u, v, data in G.edges(data=True)])

# # Determine i, j, and weights
# i = np.where(edges[:, 0] > edges[:, 1], edges[:, 1], edges[:, 0])
# j = np.where(edges[:, 0] > edges[:, 1], edges[:, 0], edges[:, 1])
# weights = edges[:, 2]


# np.savetxt(root+'sparse_i.txt', np.array(i), delimiter='\n')
# np.savetxt(root+'sparse_j.txt', np.array(j), delimiter='\n')
# np.savetxt(root+'sparse_w.txt', np.array(weights), delimiter='\n')

# level_edges = dict()
# for u, v, data in G.edges(data=True):
#     level = int(str(data['combined_score']*10)[0])
#     if level in level_edges:
#         level_edges[level].append([u, v])
#     else:
#         level_edges[level]=[[u, v]]

# for level in level_edges:
#     edges = np.array(level_edges[level])
#     sparse_i = np.where(edges[:, 0] > edges[:, 1], edges[:, 1], edges[:, 0])
#     sparse_j = np.where(edges[:, 0] > edges[:, 1], edges[:, 0], edges[:, 1])
#     np.savetxt(root+'level_'+str(level)+'_sparse_i.txt', np.array(sparse_i), delimiter='\n')
#     np.savetxt(root+'level_'+str(level)+'_sparse_j.txt', np.array(sparse_j), delimiter='\n')