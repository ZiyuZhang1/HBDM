import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.model_selection import KFold
####loadfile
with open(r'D:\study\thesis\project\HBDM-main\data\datasets\ppi\ppi_aliases2stringId.pkl', 'rb') as f:
    ppi_name2stringId = pickle.load(f)
with open(r'D:\study\thesis\project\HBDM-main\data\datasets\ppi\ppi_name2stringId.pkl', 'rb') as file:
    name2stringId = pickle.load(file)
with open('D:/study/thesis/project/HBDM-main/data/datasets/st/ppi_index.pkl', 'rb') as f:
    st_dict = pickle.load(f)   
####

# Define the root directory

compare_models = ['Dataset-ppi--RE-True--W-True--Epochs-15000--D-6--RH-25--LR-0.1--LP-False--CUDA-True',
                  'Dataset-ppi--RE-True--W-True--Epochs-15000--D-7--RH-25--LR-0.1--LP-False--CUDA-True',
                  'Dataset-st--RE-True--W-True--Epochs-15000--D-4--RH-25--LR-0.1--LP-False--CUDA-True',
                  'Dataset-stppi--RE-True--W-True--Epochs-15000--D-4--RH-25--LR-0.1--LP-False--CUDA-True'
]
##### functions
def convert_stringId(alias):
    try:
        stringId = name2stringId[alias]
    except:
        #print(alias, 'can\'t be converted by name2stringId! Now trying aliases2stringId.')
        try:
            stringId = ppi_name2stringId[alias]
        except:
            #print(alias, 'can\'t be converted by aliases2stringId! Now return None.')
            stringId = None
    #print(alias, stringId)
    return stringId
def folder_check(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        pass
def get_disease_genes(disease_df,tissue_genes):
    group_node = set()
    for gene in disease_df['Gene']:
        if gene in names:
            stringid = convert_stringId(gene)
            if stringid in tissue_genes:
                if stringid in network_genes:         
                    node = value_to_index_mapping[stringid]
                    group_node.add(node)
    return list(group_node)
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
def test_f1_auc(file_name,group_node):
    # Define your range of k values and leaf_size values
    k_values = range(2,10)

    # Split your data into 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    with open(file_name, "w") as f:
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
                random_nodes = []
                for i in train_nodes:
                    given_point = df[df['node']==i][[col for col in df.columns if col.endswith('d')]].to_numpy()
                    # Perform a k-NN search to find the k+1 nearest neighbors
                    distances, indices = kdtree.query(given_point, k=k)
                    start += (k-1)*[i]
                    dist += distances.reshape(-1).tolist()[1:]
                    neighbor += indices.reshape(-1).tolist()[1:]
                # random_nodes = random.sample(list(set(df['node'].tolist())-set(train_nodes)), (k-1))

                neighbor_df = pd.DataFrame({'start': start, 'neighbor': neighbor, 'distance': dist})
                neighbor_df = neighbor_df[~neighbor_df['neighbor'].isin(train_nodes)]
                predict_df = neighbor_df['neighbor'].value_counts().to_frame()
                predict_df.reset_index(inplace=True)
                predict_df['true'] = predict_df.apply(lambda row: 1 if row['neighbor'] in test_nodes else 0, axis=1)
                predicted_positives = predict_df['count']
                true_positives = predict_df['true']
                results.append(results_f1_aucpr(predicted_positives,true_positives))
            results = np.array(results)
            f1 = np.mean(results[:, 0])
            auc = np.mean(results[:, 1])
            # print(f"k={k-1}, leaf_size={20}",'\t',"F1:", f1,'\t',"AUC_PR:", auc)
            print(f"k={k-1}, leaf_size={20}\tF1: {f1}\tAUC_PR: {auc}", file=f)
#####
### get node and 1d,2d

for name in compare_models:
    print(name)
    root = 'D:/study/thesis/project/HBDM-main/ppi_results/models/'+name
    dataset = name.split('--')[0].split('-')[1]
    file_path_d = 'D:/study/thesis/project/HBDM-main/ppi_results/latent/'+name +'.pkl'
    with open(file_path_d, 'rb') as file:
        loaded_tensor = pickle.load(file)

    tensor = loaded_tensor.cpu()
    node_rep = tensor.detach().numpy()


    df_latent = pd.DataFrame()
    for d in range(node_rep.shape[1]):
        col_name = str(d+1)+'d'
        df_latent[col_name] = node_rep.T[d]
    df_latent['node'] = df_latent.index
    df = df_latent


    with open('D:/study/thesis/project/HBDM-main/data/datasets/'+dataset+'/ppi_index.pkl', 'rb') as f:
        value_to_index_mapping = pickle.load(f)

    names = set(ppi_name2stringId.keys())
    tissue_genes = set(st_dict.keys())
    network_genes = set(value_to_index_mapping.keys())

    disease_root = r'D:\study\thesis\project\HBDM-main\data\disease'
    output_root = r'D:\study\thesis\project\HBDM-main\ppi_results\test_results'
    disease_list = ['Atherosclerosis.tsv','Cardiovascular_Diseases.tsv','Coronary_artery_disease.tsv','Coronary_heart_disease.tsv','Myocardial_Infarction.tsv']
    
    for filename in disease_list:
        if filename.endswith('txt'):
            disease_df = pd.read_csv(os.path.join(disease_root,filename))
        elif 'drug' in filename:
            disease_df = pd.read_csv(os.path.join(disease_root,filename),sep='\t')
            disease_df['Gene']=disease_df['symbol']
        elif filename.endswith('tsv'):
            disease_df = pd.read_csv(os.path.join(disease_root,filename),sep='\t')
        folder_path = os.path.join(output_root,filename.split('.')[0])
        folder_check(folder_path)
        group_node = get_disease_genes(disease_df,tissue_genes)
        print(filename.split('.')[0], len(group_node))
        file_name = os.path.join(folder_path,name+'.txt')
        test_f1_auc(file_name,group_node)