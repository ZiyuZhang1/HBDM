#!/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/hbdm_env2/bin/python3.8
#SBATCH -J ppidata
#SBATCH -o ppidata.log
#SBATCH -e ppidata.err
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=20G
import numpy as np
import pandas as pd
import pickle
import networkx as nx

local_stringdb = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/data/string/lfs-stringdb/'
caddf = pd.read_csv('/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/HBDM/data/disease/DIS_CAD.tsv',sep='\t')

root = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/HBDM/data/datasets/ppi/'
file_path = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/HBDM/data/datasets/ppi/ppi_index.pkl'
disgenet_save = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/HBDM/data/test/cad_node.pkl'

# load local STRING database and names
df = pd.read_csv(local_stringdb+'9606.protein.info.v12.0.txt', sep='\t', header=0, usecols=['#string_protein_id', 'preferred_name'])
df['preferred_name'] = df['preferred_name'].str.upper()
stringId2name = df.set_index('#string_protein_id')['preferred_name'].to_dict()
name2stringId = df.set_index('preferred_name')['#string_protein_id'].to_dict()
df = pd.read_csv(local_stringdb+'9606.protein.aliases.v12.0.txt', sep='\t', header=0, usecols=['#string_protein_id', 'alias']).drop_duplicates(['alias'], keep='first')
df['alias'] = df['alias'].str.upper()
aliases2stringId = df.set_index('alias')['#string_protein_id'].to_dict()
  
graph_df = pd.read_csv(local_stringdb+'9606.protein.physical.links.detailed.v12.0.txt', sep=' ', header=0).convert_dtypes().replace(0, float('nan'))
graph_df = graph_df[['protein1', 'protein2','combined_score']]
G = nx.from_pandas_edgelist(graph_df, source='protein1', target='protein2', edge_attr='combined_score', create_using=nx.Graph)
print(nx.is_connected(G))
components = list(nx.connected_components(G))
# Print information about each connected component
for i, component in enumerate(components):
    print(f"Component {i + 1}")
 
    # Extract the edges for each component
    subgraph = G.subgraph(component)
    component_edges = subgraph.edges()
    print('nodes',len(subgraph.nodes),'Edges:',len(subgraph.edges))
subgraph = G.subgraph(components[0])
graph_df = nx.to_pandas_edgelist(subgraph, source='protein1', target='protein2')
proteins = sorted(list(set(graph_df['protein1'].tolist())|set(graph_df['protein2'].tolist())))
gene2node = {value: index for index, value in enumerate(proteins)}
 

# Serialize and save the Tensor to the file
with open(file_path, 'wb') as file:
    pickle.dump(gene2node, file)
# Close the file
file.close()
graph_df['node1']=graph_df['protein1'].map(gene2node)
graph_df['node2']=graph_df['protein2'].map(gene2node)
G = nx.from_pandas_edgelist(graph_df, source='node1', target='node2', edge_attr='combined_score', create_using=nx.Graph)

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
protein_names = list(aliases2stringId.keys())
protein_names.extend(list(name2stringId.keys()))
ppi_index = gene2node
node2string = {value: key for key, value in ppi_index.items()}
humans = set(ppi_index.keys())
 
cadlist = caddf['Gene'].tolist()
group_node = []
for gene in cadlist:
    if gene in protein_names:
        stringid = convert_stringId(gene)
        if stringid in humans:
            node = ppi_index[stringid]
            group_node.append(node)
with open(disgenet_save, 'wb') as file:
    pickle.dump(group_node, file)
# Close the file
file.close()
 
edges = np.array([(u, v, data['combined_score']) for u, v, data in G.edges(data=True)])
 
# Determine i, j, and weights
i = np.where(edges[:, 0] > edges[:, 1], edges[:, 1], edges[:, 0])
j = np.where(edges[:, 0] > edges[:, 1], edges[:, 0], edges[:, 1])
weights = edges[:, 2]
weights = weights*0.001
 

np.savetxt(root+'sparse_i.txt', np.array(i), delimiter='\n')
np.savetxt(root+'sparse_j.txt', np.array(j), delimiter='\n')
np.savetxt(root+'sparse_w.txt', np.array(weights), delimiter='\n')
 
weights = (weights*0.01).astype(int)
np.savetxt(root+'sparse_10.txt', np.array(weights), delimiter='\n')
 
level_edges = dict()
for u, v, data in G.edges(data=True):
    level = int(str(data['combined_score'])[0])
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