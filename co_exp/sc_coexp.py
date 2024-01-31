#!/novo/users/gzn/venepython3.9gzn/bin/python3.9
#SBATCH -J gzn_sc
#SBATCH -o gzn_sc.log
#SBATCH -e gzn_sc.err
#SBATCH --partition=compute
#SBATCH --time=1:00:00
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
    #__________________
    exp_dat = exp_dat.iloc[:, :300]
    #___________________
    exp_dat[exp_dat < threshold] = 0
    # Create an empty array with a specified shape (e.g., 0 rows, len(exp_dat.columns)) columns)
    matrix = np.empty((0, len(exp_dat.columns)))
    for gene in exp_dat.columns:
        temp_array = coexp_list(exp_dat, gene)
        # Append the new row to the empty array
        matrix = np.append(matrix, [temp_array], axis=0)
    return matrix , exp_dat.columns

## scdata
scpath = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/data/scrnaseq/scVI_Atherosclerosis_Atlas_V2.h5ad'
sc_dat = sc.read_h5ad(scpath)
sc.pp.filter_cells(sc_dat, min_genes=200)
sc.pp.filter_genes(sc_dat, min_cells=3)
sc_dat = sc_dat.raw.to_adata()
#sc_dat.obs['Disease'].unique()
# get disease data
disease_scdat = sc_dat[sc_dat.obs['Disease'] != 'Healthy',]
# normalize
#disease_scdat = disease_scdat.raw.to_adata()
disease_scdat.var_names_make_unique() 
sc.pp.normalize_total(disease_scdat, target_sum=1e4)

# local_stringdb = os.path.join('/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/data/string/lfs-stringdb/')
# df = pd.read_csv(local_stringdb+'9606.protein.info.v12.0.txt', sep='\t', header=0, usecols=['#string_protein_id', 'preferred_name'])
# df['preferred_name'] = df['preferred_name'].str.upper()
# stringId2name = df.set_index('#string_protein_id')['preferred_name'].to_dict()
# name2stringId = df.set_index('preferred_name')['#string_protein_id'].to_dict()

# stringnames = set(list(name2stringId.keys()))

# disease_scdat = disease_scdat[:, disease_scdat.var_names.isin(list(stringnames))]

start_time = time.time()
threshold = 1
sc_all_cell_coxp_matrix = coexp_matrix(disease_scdat,threshold)

end_time = time.time()

running_time = end_time - start_time

print("Running time: ", running_time, " seconds")

with open('/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/co_exp/sc_v1.pkl', 'wb') as f:
    pickle.dump(sc_all_cell_coxp_matrix, f)

