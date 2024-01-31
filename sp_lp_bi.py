#!/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/hbdm_env2/bin/python3.8
#SBATCH -J sp_lpbi
#SBATCH -o sp_lpbi.log
#SBATCH -e sp_lpbi.err
#SBATCH --partition=compute
#SBATCH --time=250:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=20G

import networkx as nx
import numpy as np
import pickle
from sklearn import metrics
from multiprocessing import Pool, freeze_support
import multiprocessing as mp
 


 
def calculate_shortest_path(args):
    G, source, target = args
    try:
        return 1/(nx.shortest_path_length(G, source=source, target=target))
    except nx.NetworkXNoPath:
        return 0
 
if __name__ == '__main__':
    freeze_support()
    for index, dataset in enumerate(['ppi_linkpredict','ppi_linkpredict2','ppi_linkpredict3','ppi_linkpredict4','ppi_linkpredict5']):
        root = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/HBDM/data/datasets/'+dataset+'/'
        record_path = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/HBDM/results/sp/sp_lp_bi'+str(index+1)+'.pkl'
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
        # Set the number of processes
        num_processes = mp.cpu_count()  # You can adjust this based on your system's capabilities
        args_list = [(G, source, target) for source, target in zip(test_i, test_j)]
    
        # Use multiprocessing Pool to parallelize the computation
        with Pool(num_processes) as pool:
            predict_label = pool.map(calculate_shortest_path, args_list)
    
        # Now, the predict_label list contains the results
        precision, recall, thresholds = metrics.precision_recall_curve(true_label,predict_label)
        roc,pr= metrics.roc_auc_score(true_label,predict_label),metrics.auc(recall,precision)
        print(roc,pr)
        with open(record_path, 'wb') as file:
            pickle.dump([roc, pr], file)
        # Close the file
        file.close()