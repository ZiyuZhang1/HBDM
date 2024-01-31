#!/novo/users/gzn/venepython3.9gzn/bin/python3.9
#SBATCH -J gzn_stv2
#SBATCH -o gzn_stv2.log
#SBATCH -e gzn_stv2.err
#SBATCH --partition=compute
#SBATCH --time=144:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=80G

import scanpy as sc
import pandas as pd
import numpy as np
import pickle
import time
import os
import multiprocessing as mp

# create co-exp matrix
def coexp_list(data, goi):
    res_array = data.multiply(data[goi], axis=0)
    coexp_count = np.count_nonzero(res_array, axis=0)
    coexp_frac = coexp_count/len(data)
    return coexp_frac

## scdata
scpath = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/data/ST/ST_plaque_all.h5ad'
sc_dat = sc.read_h5ad(scpath)
sc.pp.filter_cells(sc_dat, min_genes=200)
sc.pp.filter_genes(sc_dat, min_cells=3)
#sc_dat.obs['Disease'].unique()
# get disease datasc_dat
disease_scdat = sc_dat[sc_dat.obs['condition'] == 'Plaque',]
# normalize
disease_scdat.var_names_make_unique() 
sc.pp.normalize_total(disease_scdat, target_sum=1e4)

start_time = time.time()
threshold = 1

exp_dat = pd.DataFrame(disease_scdat.X.toarray(), columns=disease_scdat.var_names)
exp_dat[exp_dat < threshold] = 0
num_processes = mp.cpu_count()
print(num_processes)
with mp.Pool(num_processes) as pool:
    temp_arrays = pool.starmap(coexp_list, [(exp_dat, gene) for gene in exp_dat.columns])

sc_all_cell_coxp_matrix = np.vstack(temp_arrays)
final_results = (sc_all_cell_coxp_matrix,exp_dat.columns.tolist())


end_time = time.time()

running_time = end_time - start_time

print("Running time: ", running_time, " seconds")

with open('/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/co_exp/st_v2.pkl', 'wb') as f:
    pickle.dump(final_results, f)