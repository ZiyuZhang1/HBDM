#!/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/hbdm_env2/bin/python3.8
#SBATCH -J enrich
#SBATCH -o enrich.log
#SBATCH -e enrich.err
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=10G

import gseapy as gp

gene_list = ['CELSR2','PSRC1','SORT1','GOLPH3L','CTSS','IL6R']

enr = gp.enrichr(gene_list=gene_list, # or "./tests/data/gene_list.txt",
                 gene_sets=['MSigDB_Hallmark_2020','KEGG_2021_Human'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )
print(enr.results['P-value'],enr.results['Combined Score'])



# import requests ## python -m pip install requests 
# import json

# string_api_url = "https://version-11-5.string-db.org/api"
# output_format = "json"
# method = "enrichment"


# ##
# ## Construct the request
# ##

# request_url = "/".join([string_api_url, output_format, method])

# ##
# ## Set parameters
# ##

# my_genes = ['CELSR2','PSRC1','SORT1','GOLPH3L','CTSS','IL6R']

# params = {
#     "identifiers" : "%0d".join(my_genes), # your protein
#     "species" : 9606, # species NCBI identifier 
# }


# response = requests.post(request_url, data=params)

# data = json.loads(response.text)

# for row in data:
#     term = row["term"]
#     preferred_names = ",".join(row["preferredNames"])
#     fdr = float(row["fdr"])
#     description = row["description"]
#     category = row["category"]

#     if category == "Process" and fdr < 0.01:

#         ## print significant GO Process annotations

#         print("\t".join([term, preferred_names, str(fdr), description]))