#!/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/hbdm_env2/bin/python3.8
#SBATCH -J sp_l
#SBATCH -o sp_level.log
#SBATCH -e sp_level.err
#SBATCH --partition=compute
#SBATCH --time=10:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
import networkx as nx
import itertools
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import pickle
dataset = 'ppi_sc_st_4'
######## define label
pos_lists = []
root= '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/data/'
df = pd.read_csv(root+dataset+'.csv')
disease_root = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/HBDM/data/disease/'
disease_source = 'DIS_CAD'
diseasedf = pd.read_csv(disease_root+disease_source+'.tsv',sep='\t')
pos_genes = diseasedf['Gene'].to_list()
pos_lists.append(pos_genes)
disease_source = 'OT_CAD'
diseasedf = pd.read_csv(disease_root+disease_source+'.tsv',sep='\t')
pos_genes = diseasedf['symbol']
pos_lists.append(pos_genes)
disease_source = 'cad_literature'
diseasedf = pd.read_csv(disease_root+disease_source+'.csv')
pos_genes = diseasedf['gene']
pos_lists.append(pos_genes)
genticpath = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/310_targets/CAD-Genetic-Target-Discovery-2023_010323.xlsx'
geneticdf=pd.read_excel(genticpath,sheet_name='MASTER')
pos_genes = geneticdf['Gene']
pos_lists.append(pos_genes)

local_stringdb = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/data/string/lfs-stringdb/'
# load local STRING database and names
df = pd.read_csv(local_stringdb+'9606.protein.info.v12.0.txt', sep='\t', header=0, usecols=['#string_protein_id', 'preferred_name'])
df['preferred_name'] = df['preferred_name'].str.upper()
stringId2name = df.set_index('#string_protein_id')['preferred_name'].to_dict()
name2stringId = df.set_index('preferred_name')['#string_protein_id'].to_dict()
df = pd.read_csv(local_stringdb+'9606.protein.aliases.v12.0.txt', sep='\t', header=0, usecols=['#string_protein_id', 'alias']).drop_duplicates(['alias'], keep='first')
df['alias'] = df['alias'].str.upper()
aliases2stringId = df.set_index('alias')['#string_protein_id'].to_dict()

network = pd.read_csv(local_stringdb+'9606.protein.physical.links.detailed.v12.0.txt', sep=' ', header=0).convert_dtypes().replace(0, float('nan'))
# Define the reversed bin edges and labels
bins = [149, 199, 299, 399, 499, 599, 699, 799, 899, 999]
labels = [9, 8, 7, 6, 5, 4, 3, 2, 1]
#labels = [1,2,3,4,5,6,7,8,9]

# Use pd.cut to categorize the data into levels
network['weight'] = pd.cut(network['combined_score'], bins=bins, labels=labels, include_lowest=True)



def convert_stringId(alias):
    try:
        stringId = name2stringId[alias]
    except:
        #print(alias, 'can\'t be converted by name2stringId! Now trying aliases2stringId.')
        try:
            stringId = aliases2stringId[alias]
        except:
            #print(alias, 'can\'t be converted by aliases2stringId! Now return None.')
            stringId = None
    #print(alias, stringId)
    return stringId

G = nx.from_pandas_edgelist(network, source='protein1', target='protein2', edge_attr='weight', create_using=nx.Graph)


for pos_genes in pos_lists:
    print(len(pos_genes))
    pos_genesid = list(set(list(map(convert_stringId,pos_genes))))
    pos_genesid = list(set(pos_genesid)&set(G.nodes))
    len(pos_genesid)
    for threshold in [3,4,5,6,7,8,9,10]:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        pre = []
        cov = []
        for train_index, test_index in kf.split(pos_genesid):
            train = [pos_genesid[i] for i in train_index]
            test = [pos_genesid[i] for i in test_index]
            pair_combinations = list(itertools.combinations(train, 2))
            neibors = dict()
            for pairs in pair_combinations:
                path = nx.shortest_path(G, pairs[0], pairs[1],weight='weight')
                if len(path) > threshold:
                    continue
                else:
                    for gene in path[1:-1]:
                        if gene in neibors:
                            neibors[gene] += 1
                        else:
                            neibors[gene] = 1
            tp_dict = {key: neibors[key] for key in neibors.keys() if key in test}
            pre.append(sum(tp_dict.values())/sum(neibors.values()))                
            cov.append(len(tp_dict.keys())/len(test))
        print(sum(pre) / len(pre),sum(cov) / len(cov))