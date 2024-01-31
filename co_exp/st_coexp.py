#!/novo/users/gzn/venepython3.9gzn/bin/python3.9
#SBATCH -J gzn_st_T3
#SBATCH -o gzn_st_T3.log
#SBATCH -e gzn_st_T3.err
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=80G

import scanpy as sc
import pandas as pd
import numpy as np
import pickle
import time
import os



# create co-exp matrix
def coexp_list(data, goi):
    res_array = data.multiply(data[goi], axis=0)
    coexp_count = np.count_nonzero(res_array, axis=0)
    coexp_frac = coexp_count/len(data)
    return coexp_frac
def coexp_matrix(adata, threshold):
    exp_dat = pd.DataFrame(adata.X.toarray(),columns=adata.var_names)
    exp_dat[exp_dat < threshold] = 0
    # Create an empty array with a specified shape (e.g., 0 rows, len(exp_dat.columns)) columns)
    matrix = np.empty((0, len(exp_dat.columns)))
    for gene in exp_dat.columns:
        temp_array = coexp_list(exp_dat, gene)
        # Append the new row to the empty array
        matrix = np.append(matrix, [temp_array], axis=0)
    return matrix , exp_dat.columns

## stdata
stpath = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/data/ST/ST_plaque_all.h5ad'
st_dat = sc.read_h5ad(stpath)
#st_dat.obs['Disease'].unique()
# get disease data
disease_stdat = st_dat[st_dat.obs['condition'] == 'Plaque',]
# normalize
#disease_stdat = disease_stdat.raw.to_adata()
disease_stdat.var_names_make_unique() 
sc.pp.normalize_total(disease_stdat, target_sum=1e4)

local_stringdb = os.path.join('/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/data/string/lfs-stringdb/')
df = pd.read_csv(local_stringdb+'9606.protein.info.v12.0.txt', sep='\t', header=0, usecols=['#string_protein_id', 'preferred_name'])
df['preferred_name'] = df['preferred_name'].str.upper()
stringId2name = df.set_index('#string_protein_id')['preferred_name'].to_dict()
name2stringId = df.set_index('preferred_name')['#string_protein_id'].to_dict()

stringnames = set(list(name2stringId.keys()))

disease_stdat = disease_stdat[:, disease_stdat.var_names.isin(list(stringnames))]


start_time = time.time()
threshold = 3
st_all_cell_coxp_matrix = coexp_matrix(disease_stdat,threshold)

with open('/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/co_exp/T3_st_all_cell_coxp_matrix.pkl', 'wb') as f:
    pickle.dump(st_all_cell_coxp_matrix, f)

end_time = time.time()

running_time = end_time - start_time

print("Running time: ", running_time, " seconds")