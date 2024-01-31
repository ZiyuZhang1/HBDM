#!/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/hbdm_env2/bin/python3.8
#SBATCH -J sp_10
#SBATCH -o sp_10.log
#SBATCH -e sp_10.err
#SBATCH --partition=compute
#SBATCH --time=250:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G

import networkx as nx
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.neighbors import KDTree
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import metrics
import multiprocessing as mp
 
########## load network (continuous, 10level)
root = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/HBDM/data/datasets/ppi/'
record_path = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/HBDM/results/sp/sp_10_cad.pkl'
temp_neighbor_path = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/HBDM/results/sp/sp_10_cad.csv'
with open('/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/HBDM/data/test/cad_node.pkl', 'rb') as file:
    group_node = pickle.load(file) 

sparse_i = np.loadtxt(root+'sparse_i.txt')
sparse_j = np.loadtxt(root+'sparse_j.txt')
sparse_w = np.loadtxt(root+'sparse_w.txt')
 
sparse_i = sparse_i.astype(int)
sparse_j = sparse_j.astype(int)
sparse_w = sparse_w.astype(float)
 
transformed_w = 10-(sparse_w*10).round()
edges_and_weights = zip(sparse_i, sparse_j,transformed_w)
G = nx.DiGraph()
# Use from_edgelist to directly create the graph from edges
G.add_weighted_edges_from(edges_and_weights)

# bins = [149, 199, 299, 399, 499, 599, 699, 799, 899, 999]
# # labels = [9, 8, 7, 6, 5, 4, 3, 2, 1]
# labels = [100,200,300,400,500,600,700,800,900]
 
# # Use pd.cut to categorize the data into levels
# network['10level'] = pd.cut(network['combined_score'], bins=bins, labels=labels, include_lowest=True)
# network['10level'] = network['10level'].apply(lambda x: -np.log(x/1000))
# levelg = nx.from_pandas_edgelist(network, source='node1', target='node2', edge_attr='10level', create_using=nx.Graph)

ks = [3,5]
group_node = group_node[:100]

def get_neighbor_list(args):
    start_point, G, weight, k_max = args
 
    shortest_paths_weighted = nx.shortest_path_length(G, source=start_point, weight = weight)
    start=k_max * [start_point]
 
    neighborlist = list(shortest_paths_weighted.keys())
    neighborlist.pop(0)
    neighbor=neighborlist[:k_max]
 
    distancelist = list(shortest_paths_weighted.values())
    distancelist.pop(0)
    dist=distancelist[:k_max]
    return start, neighbor, dist
 
def parallel_worker(args):
    return get_neighbor_list(args)
 
if __name__ == "__main__":
    num_processes = mp.cpu_count()
    mp.freeze_support()
 
    pool = mp.Pool(processes=num_processes)
    args_list = [(node, G, 'weight', max(ks)) for node in group_node]
   
    results = pool.map(parallel_worker, args_list)
    pool.close()
    pool.join()
 
    starts = []
    neighbors = []
    dists = []
 
    for result in results:
        start, neighbor, dist = result
        starts.extend(start)
        neighbors.extend(neighbor)
        dists.extend(dist)
    neighbor_df = pd.DataFrame({'start': starts, 'neighbor': neighbors, 'distance': dists})
 
    neighbor_df.to_csv(temp_neighbor_path,index=False)
 
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(group_node):
        train_nodes = [group_node[i] for i in train_index]
        test_nodes = [group_node[i] for i in test_index]
        kfresults = []
 
        kfneighbor_df = neighbor_df[neighbor_df['start'].isin(train_nodes)]
 
        groups = kfneighbor_df.groupby('start')
        for k in ks:
            results = []
            for i, (key, subdf) in enumerate(groups):
                subdf = subdf.head(k)
                if i == 0:
                    kneighbor_df = subdf
                else:
                    kneighbor_df = pd.concat([kneighbor_df, subdf], ignore_index=True)
            kneighbor_df = kneighbor_df[~kneighbor_df['neighbor'].isin(train_nodes)]
            predict_df = kneighbor_df['neighbor'].value_counts().to_frame()
            predict_df.reset_index(inplace=True)
            predict_df.rename(columns={'neighbor':'count','index':'neighbor'},inplace=True)
            predict_df['true'] = predict_df.apply(lambda row: 1 if row['neighbor'] in test_nodes else 0, axis=1)
            if len(predict_df[predict_df['true']==1])==0:
                results.append([0, 0])
            else:
                predicted_positives = predict_df['count']
                true_positives = predict_df['true']
                precision, recall, thresholds = metrics.precision_recall_curve(true_positives,predicted_positives)
                roc,pr= metrics.roc_auc_score(true_positives,predicted_positives),metrics.auc(recall,precision)
                results.append([roc,pr])
        kfresults.append(results)
    kfresults = np.array(kfresults)
    roc_ks = np.mean(kfresults[:, 0::2], axis=0)
    pr_ks = np.mean(kfresults[:, 1::2], axis=0)
 
    with open(record_path, 'wb') as file:
        pickle.dump([roc_ks, pr_ks], file)
    # Close the file
    file.close()

# ########## load network (continuous, 10level)
# network = pd.read_csv('/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/HBDM/data/ppi_connect.csv')

# G = nx.from_pandas_edgelist(network, source='node1', target='node2', create_using=nx.Graph)

# ############ experirmnt (complex detection)
# with open('/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/HBDM/data/test/complex/human_complexes_node.pkl', 'rb') as file:
#     complexs_id = pickle.load(file)

 
# precision_complexs = []
# recall_complexs = []
# for complex_name in complexs_id:
#     group_node = complexs_id[complex_name]
#     precision_ks = []
#     recall_ks = []
#     for i in group_node:
#         shortest_paths_weighted = nx.shortest_path_length(G, source=i)
#         neighborlist = list(shortest_paths_weighted.keys())
#         neighborlist.pop(0)
#         test_nodes = list(set(group_node)-set([i]))
#         precision = []
#         recall = []
#         for top in [2,3,4,5,6,7,8,9,10]:
#             neighbor = neighborlist[:top]
#             tp = len(set(neighbor)&set(test_nodes))
#             fp = len(neighbor)-tp
#             fn = len(test_nodes)-tp
 
#             precision.append(tp/(tp+fp))
#             recall.append(tp/(tp+fn))
#         precision_ks.append(precision)
#         recall_ks.append(recall)
   
#     precision_complexs.append(np.array(precision_ks).mean(axis=0))
#     recall_complexs.append(np.array(recall_ks).mean(axis=0))
 
# avg_precision = np.array(precision_complexs).mean(axis=0)
# avg_recall = np.array(recall_complexs).mean(axis=0)
# avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)

# file_path = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/HBDM/results/sp/sp_com_bi_f1.pkl'
# # Serialize and save the Tensor to the file
# with open(file_path, 'wb') as file:
#     pickle.dump([precision_complexs, recall_complexs, avg_f1], file)
# # Close the file
# file.close()