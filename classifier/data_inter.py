import pandas as pd
import numpy as np
import pickle

def get_df(dataset):
    files_path = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/HBDM/results/models/Dataset-'+dataset+'--RE-True--W-True--Epochs-15000--D-4--RH-25--LR-0.1--LP-False--CUDA-True/'
    path1 = files_path + 'latent.pkl'
    path2 = files_path + 'RE.pkl'

    with open(path1, 'rb') as file:
        latent = pickle.load(file)
    latent = np.array(latent)
    with open(path2, 'rb') as file:
        re = pickle.load(file)
    re = np.array(re)

    convertpath = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/HBDM/data/datasets/'+dataset+'/ppi_index.pkl'
    with open(convertpath, 'rb') as file:
        ppi_index = pickle.load(file)
    data = np.concatenate((latent, re[:, np.newaxis]), axis=1)
    df_latent = pd.DataFrame()
    for i, d in enumerate(range(data.shape[1])):
        if i == len(range(data.shape[1]))-1:
            col_name = 're'
        else:
            col_name = str(d+1)+'d'
        df_latent[col_name] = data.T[d]
    df_latent['node'] = df_latent.index
    inv_dict = {v: k for k, v in ppi_index.items()}
    df_latent = df_latent.add_prefix(dataset+'_')
    df_latent['gene'] = df_latent[dataset+'_node'].map(inv_dict)
    df = df_latent.loc[:, ~df_latent.columns.str.endswith('node')]
    return df

ppidf = get_df('ppi')
scdf = get_df('sc')
stdf = get_df('st')
df = pd.merge(ppidf,scdf,on='gene')
df = pd.merge(df,stdf,on='gene')
print(len(ppidf),len(scdf),len(stdf),len(df))

df.to_csv('/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/data/ppi_sc_st_4.csv',index=False)