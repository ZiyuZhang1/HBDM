import pandas as pd
import pickle
import numpy as np
import random
from sklearn.neighbors import KDTree
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import metrics

def complex_detection(model):
    node_rep = [tensor.detach().cpu().numpy() for tensor in model.latent_z]
    node_rep = np.array(node_rep)
    df_latent = pd.DataFrame()
    for d in range(node_rep.shape[1]):
        col_name = str(d+1)+'d'
        df_latent[col_name] = node_rep.T[d]
    df_latent['node'] = df_latent.index
    df = df_latent

    with open(r'D:\study\thesis\project\HBDM-main\data\complexes\human_complexes_node.pkl', 'rb') as file:
        complexs_id = pickle.load(file)

    k_values = [2,3,4,5,6,7,8,9,10]
    precision_complexs= []
    recall_complexs=[]

    for complex_name in complexs_id:
        group_node = complexs_id[complex_name]
        # for each eomplex, get precision and recall values of varing k
        precision_ks = []
        recall_ks = []

        ## get final df: node, cluster, 1d, 2d, label
        df['label'] = df['node'].apply(lambda x: 1 if x in group_node else 0)
        # Set the index to match the values in column 'node'
        df = df.set_index('node')
        # Reset the index to its default integer index
        df = df.reset_index()    
        # with open(output_file, "w") as f:
        for k in k_values:
            k+=1

            kdtree = KDTree(df[[col for col in df.columns if col.endswith('d')]].to_numpy(), leaf_size=20)
            ## the pr list contains all participant results for one knn
            precision = []
            recall = []
            for i in group_node:
                test_nodes = list(set(group_node)-set([i]))
                given_point = df[df['node']==i][[col for col in df.columns if col.endswith('d')]].to_numpy()
                # Perform a k-NN search to find the k+1 nearest neighbors
                distances, indices = kdtree.query(given_point, k=k)
                # start += (k-1)*[i]
                # dist += distances.reshape(-1).tolist()[1:]
                neighbor = indices.reshape(-1).tolist()[1:]

                tp = len(set(neighbor)&set(test_nodes))
                fp = len(neighbor)-tp
                fn = len(test_nodes)-tp

                precision.append(tp/(tp+fp))
                recall.append(tp/(tp+fn))
            precision_ks.append(sum(precision)/len(precision))
            recall_ks.append(sum(recall)/len(recall))
        precision_complexs.append(precision_ks)
        recall_complexs.append(recall_ks)
    
    avg_precision = np.array(precision_complexs).mean(axis=0)
    avg_recall = np.array(recall_complexs).mean(axis=0)
    avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    return avg_f1

def disgenet_detection(model):
    node_rep = [tensor.detach().cpu().numpy() for tensor in model.latent_z]
    # node_rep = np.array(node_rep)
    # df_latent = pd.DataFrame()
    # for d in range(node_rep.shape[1]):
    #     col_name = str(d+1)+'d'
    #     df_latent[col_name] = node_rep.T[d]
    # df_latent['node'] = df_latent.index
    # df = df_latent
    #######################################
    node_rep = np.array(node_rep)
    re = [tensor.detach().cpu().numpy() for tensor in model.gamma]
    re = np.array(re)
    data = np.concatenate((node_rep, re[:, np.newaxis]), axis=1)
    df_latent = pd.DataFrame()
    for d in range(data.shape[1]):
        col_name = str(d+1)+'d'
        df_latent[col_name] = data.T[d]
    df_latent['node'] = df_latent.index
    df = df_latent
    ########################################
    with open(r'D:\study\thesis\project\HBDM-main\data\disease\cad_node.pkl', 'rb') as file:
        group_node = pickle.load(file)

    # k_values = [10,20,30,40,50]
    k_values = [50,100,150,200,250,300,350,400] 
    #################################################################################################
    # pathways_id = dict(sorted(pathways_id.items(), key=lambda item: item[1], reverse=True)[:10])
    outlier = []
    #################################################################################################

    # for each eomplex, get precision and recall values of varing k
    roc_ks = []
    pr_ks = []

    ## get final df: node, cluster, 1d, 2d, label
    df['label'] = df['node'].apply(lambda x: 1 if x in group_node else 0)
    # Set the index to match the values in column 'node'
    df = df.set_index('node')
    # Reset the index to its default integer index
    df = df.reset_index()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for k in k_values:
        k+=1
        results = []

        # Build a k-d tree from the points
        kdtree = KDTree(df[[col for col in df.columns if col.endswith('d')]].to_numpy(), leaf_size=20)
        for train_index, test_index in kf.split(group_node):
            train_nodes = [group_node[i] for i in train_index]
            test_nodes = [group_node[i] for i in test_index]
            start = []
            dist = []
            neighbor = []

            for i in train_nodes:
                given_point = df[df['node']==i][[col for col in df.columns if col.endswith('d')]].to_numpy()
                # Perform a k-NN search to find the k+1 nearest neighbors
                distances, indices = kdtree.query(given_point, k=k)
                start += (k-1)*[i]
                dist += distances.reshape(-1).tolist()[1:]
                neighbor += indices.reshape(-1).tolist()[1:]


            neighbor_df = pd.DataFrame({'start': start, 'neighbor': neighbor, 'distance': dist})
            neighbor_df = neighbor_df[~neighbor_df['neighbor'].isin(train_nodes)]
            predict_df = neighbor_df['neighbor'].value_counts().to_frame()
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
        results = np.array(results)
        roc = np.mean(results[:, 0])
        pr = np.mean(results[:, 1])
        roc_ks.append(roc)
        pr_ks.append(pr)

    return roc_ks,pr_ks

def pathway_detection(model):
    node_rep = [tensor.detach().cpu().numpy() for tensor in model.latent_z]
    node_rep = np.array(node_rep)
    df_latent = pd.DataFrame()
    for d in range(node_rep.shape[1]):
        col_name = str(d+1)+'d'
        df_latent[col_name] = node_rep.T[d]
    df_latent['node'] = df_latent.index
    df = df_latent

    with open(r'D:\study\thesis\project\HBDM-main\data\pathway\pathway_node.pkl', 'rb') as file:
        pathways_id = pickle.load(file)

    k_values = [50,100,150,200,250,300]
    roc_pathways= []
    prauc_pathways=[] 
    #################################################################################################
    pathways_id = dict(sorted(pathways_id.items(), key=lambda item: item[1], reverse=True)[:10])
    pathway_outlier = []
    #################################################################################################

    for pathway in pathways_id:
        group_node = pathways_id[pathway]
        # for each eomplex, get precision and recall values of varing k
        roc_ks = []
        prauc_ks = []

        ## get final df: node, cluster, 1d, 2d, label
        df['label'] = df['node'].apply(lambda x: 1 if x in group_node else 0)
        # Set the index to match the values in column 'node'
        df = df.set_index('node')
        # Reset the index to its default integer index
        df = df.reset_index()

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for k in k_values:
            k+=1
            results = []

            # Build a k-d tree from the points
            kdtree = KDTree(df[[col for col in df.columns if col.endswith('d')]].to_numpy(), leaf_size=20)
            for train_index, test_index in kf.split(group_node):
                train_nodes = [group_node[i] for i in train_index]
                test_nodes = [group_node[i] for i in test_index]
                start = []
                dist = []
                neighbor = []

                for i in train_nodes:
                    given_point = df[df['node']==i][[col for col in df.columns if col.endswith('d')]].to_numpy()
                    # Perform a k-NN search to find the k+1 nearest neighbors
                    distances, indices = kdtree.query(given_point, k=k)
                    start += (k-1)*[i]
                    dist += distances.reshape(-1).tolist()[1:]
                    neighbor += indices.reshape(-1).tolist()[1:]


                neighbor_df = pd.DataFrame({'start': start, 'neighbor': neighbor, 'distance': dist})
                neighbor_df = neighbor_df[~neighbor_df['neighbor'].isin(train_nodes)]
                predict_df = neighbor_df['neighbor'].value_counts().to_frame()
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
            results = np.array(results)
            roc = np.mean(results[:, 0])
            pr = np.mean(results[:, 1])
            roc_ks.append(roc)
            prauc_ks.append(pr)
        roc_pathways.append(roc_ks)
        prauc_pathways.append(prauc_ks)
        if roc_ks.count(0) >= 4:
            pathway_outlier.append(pathway)
    avg_roc = np.array(roc_pathways).mean(axis=0)
    avg_pr = np.array(prauc_pathways).mean(axis=0)
    print('pathway_outliers: ',len(pathway_outlier))
    return avg_roc,avg_pr

# ############## disgenet test random
# import random
# exp = []
# for i in range(10000):
#     # Total number of nodes and positive nodes
#     total_nodes = 307*[1]+(17530-307)*[0]

#     # Number of nodes to randomly choose
#     num_nodes_chosen = 50

#     # Simulating random selection of nodes
#     randomly_chosen_nodes = random.sample(total_nodes, num_nodes_chosen)

#     # Assuming you have the true positives (TP) and false positives (FP) for the selected 50 nodes
#     # Replace these values with your actual experiment results
#     TP = randomly_chosen_nodes.count(1) # Number of true positives
#     FP = randomly_chosen_nodes.count(0)   # Number of false positives

#     # Calculating precision
#     precision = TP / (TP + FP)
#     exp.append(precision)
# sum(exp)/len(exp) # 0.017448000000000796


def results_f1_aucpr(predicted_positives,true_positives,plot=False):
    # Example usage
    precision_values, recall_values = calculate_precision_recall_curve(true_positives, predicted_positives)
    auc_pr = calculate_auc_pr(precision_values, recall_values)



    # Obtain the corresponding thresholds
    thresholds = sorted(predicted_positives, reverse=True)

    best_f1_threshold, best_f1, = find_best_thresholds(precision_values, recall_values, thresholds)

    # print("Best Threshold for F1:", best_f1_threshold)
    # print("Best F1:", best_f1)
    # print('precision here ', precision_values[best_f1_threshold])
    # print('recall here ', recall_values[best_f1_threshold])
    # print("AUC-PR:", auc_pr)
    if plot:
        # Plot the Precision-Recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall_values, precision_values, marker='o', linestyle='-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)

        line_colors = ['r', 'b', 'g', 'y']

        # Add vertical lines at i=10, i=50, i=100, and i=200 with different colors
        top_cuts = [10, 50, 100, 200]

        for i, top_cut in enumerate(top_cuts):
            color_index = i % len(line_colors)  # Cyclically select colors from the list
            plt.axvline(x=recall_values[top_cut], color=line_colors[color_index], linestyle='--', label=f'Top {top_cut}')
        # Show or save the plot
        plt.legend()  # Add a legend to explain the vertical lines
        plt.show()

    return [best_f1, auc_pr]

def calculate_precision_recall_curve(y_true, y_scores):
    # Sort the scores and true labels in descending order of scores
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_scores = np.array(y_scores)[desc_score_indices]
    y_true = np.array(y_true)[desc_score_indices]

    precision_values = []
    recall_values = []
    num_true_positives = 0
    num_predicted_positives = 0
    total_true_positives = sum(y_true)
    
    for i in range(len(y_true)):
        num_predicted_positives += 1
        if y_true[i] == 1:
            num_true_positives += 1
        precision = num_true_positives / num_predicted_positives
        recall = num_true_positives / total_true_positives
        precision_values.append(precision)
        recall_values.append(recall)

    return precision_values, recall_values

def calculate_auc_pr(precision, recall):
    return np.trapz(precision, recall)
def find_best_thresholds(precision, recall, thresholds):
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    best_precision_threshold = thresholds[0]
    best_recall_threshold = thresholds[0]
    best_f1_threshold = thresholds[0]
    for i in range(len(thresholds)):
        if precision[i] + recall[i] != 0:
            f1 = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

            if precision[i] > best_precision:
                best_precision = precision[i]
                # best_precision_threshold = thresholds[i]
                best_precision_threshold = i
            if recall[i] > best_recall:
                best_recall = recall[i]
                # best_recall_threshold = thresholds[i]
                best_recall_threshold=i
            if f1 > best_f1:
                best_f1 = f1
                # best_f1_threshold = thresholds[i]
                best_f1_threshold = i

    return best_f1_threshold, best_f1