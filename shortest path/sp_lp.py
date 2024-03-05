import networkx as nx 
import numpy as np
import pickle
from sklearn import metrics
from multiprocessing import Pool, freeze_support
import multiprocessing as mp

root = 'D:/study/thesis/project/HBDM-main/data/datasets/ppi_linkpredict/'
record_path = r'D:\study\thesis\project\HBDM-main\ppi_results\sp_results\sp_lp\sp_lp_bi.pkl'
sparse_i = np.loadtxt(root+'sparse_i.txt')
sparse_j = np.loadtxt(root+'sparse_j.txt')
sparse_w = np.loadtxt(root+'sparse_w.txt')

sparse_i = sparse_i.astype(int)
sparse_j = sparse_j.astype(int)
sparse_w = sparse_w.astype(float)

string_score_transform = np.vectorize(lambda x: -np.log(x/1000))
transformed_w = string_score_transform(sparse_w)

edges_and_weights = zip(sparse_i, sparse_j,transformed_w)

G = nx.DiGraph()

# Use from_edgelist to directly create the graph from edges
G.add_weighted_edges_from(edges_and_weights)

sparse_i_rem = np.loadtxt(root+'sparse_i_rem.txt')
sparse_j_rem = np.loadtxt(root+'sparse_j_rem.txt')
sparse_i_rem = sparse_i_rem.astype(int)
sparse_j_rem = sparse_j_rem.astype(int)

non_sparse_i = np.loadtxt(root+'non_sparse_i.txt')
non_sparse_j = np.loadtxt(root+'non_sparse_j.txt')
non_sparse_i = non_sparse_i.astype(int)
non_sparse_j = non_sparse_j.astype(int)

test_i = np.concatenate((sparse_i_rem, non_sparse_i))
test_j = np.concatenate((sparse_j_rem, non_sparse_j))
true_label = np.array([1]*sparse_i_rem.shape[0]+[0]*non_sparse_i.shape[0])

def calculate_shortest_path(args):
    G, source, target = args
    try:
        return 1/(nx.shortest_path_length(G, source=source, target=target))
    except nx.NetworkXNoPath:
        return 0

if __name__ == '__main__':
    freeze_support()

    root = 'D:/study/thesis/project/HBDM-main/data/datasets/ppi_linkpredict/'

    # Set the number of processes
    num_processes = mp.cpu_count()  # You can adjust this based on your system's capabilities
    args_list = [(G, source, target) for source, target in zip(test_i, test_j)]

    # Use multiprocessing Pool to parallelize the computation
    with Pool(num_processes) as pool:
        predict_label = pool.map(calculate_shortest_path, args_list)
    # with open(record_path, 'wb') as file:
    #     pickle.dump([true_label, predict_label], file)
    # Close the file
    # file.close()    
    # Now, the predict_label list contains the results
    precision, recall, thresholds = metrics.precision_recall_curve(true_label,predict_label)
    roc,pr= metrics.roc_auc_score(true_label,predict_label),metrics.auc(recall,precision)
    with open(r'D:\study\thesis\project\HBDM-main\ppi_results\sp_results\sp_lp\sp_lp_bi.pkl', 'rb') as file:
        splpbi = pickle.load(file)

    if np.all(np.array(true_label) == np.array(splpbi[0])):
        print('same true')
    if np.all(np.array(predict_label) == np.array(splpbi[1])):
        print('same pre')
    checkroc = metrics.roc_auc_score(splpbi[0],splpbi[1])
    with open(record_path, 'wb') as file:
        pickle.dump([roc, pr], file)
    # Close the file
    file.close()


