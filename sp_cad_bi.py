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
network = pd.read_csv(r'D:\study\thesis\project\HBDM-main\data\ppi_connect.csv')

string_score_transform = lambda x: -np.log(x/1000)
network['continuous'] = network['combined_score'].apply(string_score_transform)
G = nx.from_pandas_edgelist(network, source='node1', target='node2', edge_attr='continuous', create_using=nx.Graph)

# bins = [149, 199, 299, 399, 499, 599, 699, 799, 899, 999]
# # labels = [9, 8, 7, 6, 5, 4, 3, 2, 1]
# labels = [100,200,300,400,500,600,700,800,900]

# # Use pd.cut to categorize the data into levels
# network['10level'] = pd.cut(network['combined_score'], bins=bins, labels=labels, include_lowest=True)
# network['10level'] = network['10level'].apply(lambda x: -np.log(x/1000))
# levelg = nx.from_pandas_edgelist(network, source='node1', target='node2', edge_attr='10level', create_using=nx.Graph)
record_path =  r'D:\study\thesis\project\HBDM-main\data\cad_neighbors_temp_bi.pkl'
temp_neighbor_path = r'D:\study\thesis\project\HBDM-main\data\cad_neighbors_temp_bi.csv'
with open(r'D:\study\thesis\project\HBDM-main\data\disease\cad_node.pkl', 'rb') as file:
    group_node = pickle.load(file)
group_node = group_node

ks = [3,5,10,25,50,75,100]

def get_neighbor_list(args):
    start_point, G, weight, k_max = args

    shortest_paths_weighted = nx.shortest_path_length(G, source=start_point)
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
    args_list = [(node, G, 'continuous', max(ks)) for node in group_node]
    
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
    print([roc_ks,pr_ks])
    with open(record_path, 'wb') as file:
        pickle.dump([roc_ks, pr_ks], file)
    # Close the file
    file.close()
# with mp.Pool(num_processes) as pool:
#     temp_arrays = pool.starmap(get_neighbor_list, [(nodes, G,'continuous',max(ks)) for nodes in group_lists])
 


# start = []
# neighbor = []
# dist = []    
# for start_point in group_node:
#     shortest_paths_weighted = nx.shortest_path_length(G, source=start_point,weight=weight)
#     # start=k_max*[start_point]
#     start.extend(k_max*[start_point])

#     neighborlist = list(shortest_paths_weighted.keys())
#     neighborlist.pop(0)
#     # neighbor=neighborlist[:k_max]
#     neighbor.extend(neighborlist[:k_max])
                    
#     distancelist = list(shortest_paths_weighted.values())
#     distancelist.pop(0)
#     # dist=distancelist[:k_max]
#     dist.extend(distancelist[:k_max])


# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# for train_index, test_index in kf.split(group_node):
#     train_nodes = [group_node[i] for i in train_index]
#     test_nodes = [group_node[i] for i in test_index]
#     kfresults = []
#     start = []
#     neighbor = []
#     dist = []
#     for start_point in train_nodes:
#         shortest_paths_weighted = nx.shortest_path_length(G, source=start_point,weight='continuous')
#         start.extend(max(ks)*[start_point])
#         neighborlist = list(shortest_paths_weighted.keys())
#         neighborlist.pop(0)
#         neighbor.extend(neighborlist[:max(ks)])
#         distancelist = list(shortest_paths_weighted.values())
#         distancelist.pop(0)
#         dist.extend(distancelist[:max(ks)])
#     neighbor_df = pd.DataFrame({'start': start, 'neighbor': neighbor, 'distance': dist})

#     groups = neighbor_df.groupby('start')
#     for k in ks:
#         results = []
#         kneighbor_df = pd.DataFrame()
#         for key, subdf in groups:
#             subdf = subdf.head(k)
#             kneighbor_df = pd.concat([kneighbor_df, subdf], ignore_index=True)
#         kneighbor_df = kneighbor_df[~kneighbor_df['neighbor'].isin(train_nodes)]
#         predict_df = kneighbor_df['neighbor'].value_counts().to_frame()
#         predict_df.reset_index(inplace=True)
#         predict_df.rename(columns={'neighbor':'count','index':'neighbor'},inplace=True)
#         predict_df['true'] = predict_df.apply(lambda row: 1 if row['neighbor'] in test_nodes else 0, axis=1)
#         if len(predict_df[predict_df['true']==1])==0:
#             results.append([0, 0])
#         else:
#             predicted_positives = predict_df['count']
#             true_positives = predict_df['true']
#             precision, recall, thresholds = metrics.precision_recall_curve(true_positives,predicted_positives)
#             roc,pr= metrics.roc_auc_score(true_positives,predicted_positives),metrics.auc(recall,precision)
#             results.append([roc,pr])
#     kfresults.append(results)
# kfresults = np.array(kfresults)
# roc_ks = np.mean(kfresults[:, 0::2], axis=0)
# pr_ks = np.mean(kfresults[:, 1::2], axis=0)

# roc = np.mean(results[:, 0])






# pr = np.mean(results[:, 1])
# roc_ks.append(roc)
# pr_ks.append(pr)
############ experirmnt (complex detection)
# with open(r'D:\study\thesis\project\HBDM-main\data\complexes\human_complexes_node.pkl', 'rb') as file:
#     complexs_id = pickle.load(file)
# complexs_id = dict(sorted(complexs_id.items(), key=lambda item: item[1], reverse=True)[:10])

# precision_complexs = []
# recall_complexs = []
# for complex_name in complexs_id:
#     group_node = complexs_id[complex_name]
#     precision_ks = []
#     recall_ks = []
#     for i in group_node:
#         shortest_paths_weighted = nx.shortest_path_length(G, source=i,weight='continuous') 
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

# precision_complexs = []
# recall_complexs = []
# for complex_name in complexs_id:
#     group_node = complexs_id[complex_name]
#     precision_ks = []
#     recall_ks = []
#     for i in group_node:
#         shortest_paths_weighted = nx.shortest_path_length(levelg, source=0,weight='10level') 
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

# ############ experirmnt (disgenet detection)
# for pos_genes in pos_lists:
#     print(len(pos_genes))
#     pos_genesid = list(set(list(map(convert_stringId,pos_genes))))
#     pos_genesid = list(set(pos_genesid)&set(G.nodes))
#     len(pos_genesid)
#     for k in [3,4,5,6,7,8,9,10]:
#         kf = KFold(n_splits=5, shuffle=True, random_state=42)
#         pre = []
#         cov = []
#         for train_index, test_index in kf.split(pos_genesid):
#             train = [pos_genesid[i] for i in train_index]
#             test = [pos_genesid[i] for i in test_index]
#             neibors = dict()
#             for start in train:
#                 distances = nx.shortest_path_length(G, start, weight='weight')
#                 distances = {node: distance for node, distance in distances.items()}
#                 distances = dict(sorted(distances.items(), key=lambda item: item[1]))
#                 nearest_nodes = list(distances.keys())[1:k+1]
#                 for gene in nearest_nodes:
#                     if gene in neibors:
#                         neibors[gene] += 1
#                     else:
#                         neibors[gene] = 1
#             tp_dict = {key: neibors[key] for key in neibors.keys() if key in test}
#             pre.append(sum(tp_dict.values())/sum(neibors.values()))                
#             cov.append(len(tp_dict.keys())/len(test))
#         print(sum(pre) / len(pre),sum(cov) / len(cov))

# # ############ experirmnt (link prediction)
# path1 = r''
# sparse_i = np.loadtxt(path1)
# sparse_j = np.loadtxt(path2)
# sparse_i = sparse_i.astype(int)
# sparse_j = sparse_j.astype(int)
# edges = zip(sparse_i, sparse_j)