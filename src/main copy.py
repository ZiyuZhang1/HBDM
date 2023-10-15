import torch
import sys
import pickle
import os


sys.path.append('./src/')


from HBDM import LSM
    
model_name = 'Dataset-ppi--RE-True--W-True--Epochs-15000--D-9--RH-25--LR-0.1--LP-False--CUDA-True'
model_path = 'D:/study/thesis/project/HBDM-main/ppi_results/models/'+model_name+'.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)
print(model.total_K)

def is_sparse(tensor):
    return isinstance(tensor, torch.Tensor) and tensor.layout == torch.sparse_coo

root = 'D:/study/thesis/project/HBDM-main/ppi_results/models/'+model_name
os.mkdir(root)

# Specify the variables to be saved
variables_to_save = {
    'k_exp_dist': [tensor.cpu().numpy() if not is_sparse(tensor) else tensor.to_dense().cpu().numpy() for tensor in model.k_exp_dist],
    'final_idx': [tensor.cpu().numpy() for tensor in model.final_idx],
    'general_cl_id': [tensor.cpu().numpy() for tensor in model.general_cl_id],
    'general_mask': [tensor.cpu().numpy() for tensor in model.general_mask]
}

# Loop through the variables and save them with their names as file names
for var_name, var_data in variables_to_save.items():
    file_path = os.path.join(root, var_name + '.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(var_data, f)

