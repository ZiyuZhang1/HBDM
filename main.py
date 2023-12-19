
import argparse
import torch
import numpy as np
import torch.optim as optim
import sys
from tqdm import tqdm
import pickle
import os
print(sys.path)

sys.path.append('./src/')
print(sys.path)
parser = argparse.ArgumentParser(description='Hierarchical Block Distance Model')
#####for now changed 
parser.add_argument('--RE', type=eval, 
                      choices=[True, False], default=True,
                    help='activates random effects')
parser.add_argument('--W', type=eval, 
                      choices=[True, False], default=False,
                    help='activates random effects')
parser.add_argument('--epochs', type=int, default=15000, metavar='N',
                    help='number of epochs for training (default: 15K)')

####### keep

parser.add_argument('--RH', type=int, default=25, metavar='N',
                    help='number of epochs to rebuild the hierarchy from scratch (default: 25)')

parser.add_argument('--cuda', type=eval, 
                      choices=[True, False],  default=True,
                    help='CUDA training')

parser.add_argument('--LP',type=eval, 
                      choices=[True, False], default=False,
                    help='performs link prediction')

parser.add_argument('--D', type=int, default=4
                    , metavar='N',
                    help='dimensionality of the embeddings (default: 2)')

parser.add_argument('--lr', type=int, default=0.1, metavar='N',
                    help='learning rate for the ADAM optimizer (default: 0.1)')
# changed dataset
parser.add_argument('--dataset', type=str, default='ppi',
                    help='dataset to apply HBDM')





args = parser.parse_args()

if  args.cuda:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')

if args.W and args.RE:
    from HBDM_RE_W import LSM
elif args.RE:
    from HBDM_RE import LSM
elif args.W:
    from HBDM_W import LSM
else:
    from HBDM import LSM
    
    
    


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    name = f"Dataset-{args.dataset}--RE-{args.RE}--W-{args.W}--Epochs-{args.epochs}--D-{args.D}--RH-{args.RH}--LR-{args.lr}--LP-{args.LP}--CUDA-{args.cuda}"
    writer = SummaryWriter(log_dir=r"D:\study\thesis\project\HBDM-main\ppi_results\training curve\{}".format(name))
    
    latent_dims=[args.D]
    datasets=[args.dataset]
    for dataset in datasets:
        for latent_dim in latent_dims:
            if args.LP:
                # file denoting rows i of missing links, with i<j 
                sparse_i_rem=torch.from_numpy(np.loadtxt("./datasets/"+dataset+'/sparse_i_rem.txt')).long().to(device)
                # file denoting columns j of missing links, with i<j
                sparse_j_rem=torch.from_numpy(np.loadtxt("./datasets/"+dataset+'/sparse_j_rem.txt')).long().to(device)
                # file denoting negative sample rows i, with i<j
                non_sparse_i=torch.from_numpy(np.loadtxt("./datasets/"+dataset+'/non_sparse_i.txt')).long().to(device)
                # file denoting negative sample columns, with i<j
                non_sparse_j=torch.from_numpy(np.loadtxt("./datasets/"+dataset+'/non_sparse_j.txt')).long().to(device)
               
            else:
                non_sparse_i=None
                non_sparse_j=None
                sparse_i_rem=None
                sparse_j_rem=None
            if args.W:
                sparse_i = []
                sparse_j = []
                filelist = os.listdir("./data/datasets/"+dataset)
                link_w = [int(files.split('_')[1]) for files in filelist if files.startswith('level') and files.endswith('i.txt')]
                for level in link_w:
                    inputfile = "./data/datasets/"+dataset+'/level_'+str(level)
                    sparse_i_level=torch.from_numpy(np.loadtxt(inputfile+'_sparse_i.txt')).long().to(device)
                    # input data, link column positions with i<j
                    sparse_j_level=torch.from_numpy(np.loadtxt(inputfile+'_sparse_j.txt')).long().to(device)
                    sparse_i.append(sparse_i_level)
                    sparse_j.append(sparse_j_level)
                N=int(torch.cat(sparse_j).max()+1)
                model = LSM(torch.randn(N,latent_dim),sparse_i,sparse_j,link_w,N,latent_dim=latent_dim,non_sparse_i=non_sparse_i,non_sparse_j=non_sparse_j,sparse_i_rem=sparse_i_rem,sparse_j_rem=sparse_j_rem,CVflag=True,graph_type='undirected',missing_data=False,device=device,LP=args.LP).to(device)
            else:
                # input data, link rows i positions with i<j
                sparse_i=torch.from_numpy(np.loadtxt("./data/datasets/"+dataset+'/sparse_i.txt')).long().to(device)
                # input data, link column positions with i<j
                sparse_j=torch.from_numpy(np.loadtxt("./data/datasets/"+dataset+'/sparse_j.txt')).long().to(device)
                N=int(sparse_j.max()+1)
                model = LSM(torch.randn(N,latent_dim),sparse_i,sparse_j,N,latent_dim=latent_dim,non_sparse_i=non_sparse_i,non_sparse_j=non_sparse_j,sparse_i_rem=sparse_i_rem,sparse_j_rem=sparse_j_rem,CVflag=True,graph_type='undirected',missing_data=False,device=device,LP=args.LP).to(device)


                
            # Missing data here denoted if Marginalization is applied or not
            # In case missing data is set to True then the input should be the complete graph
            
            optimizer = optim.Adam(model.parameters(), args.lr)  
            elements=(N*(N-1))*0.5
            for epoch in tqdm(range(args.epochs),desc="HBDM is Running…",ascii=False, ncols=75):
                if (epoch%args.RH==0):
                    model.build_hierarchy=True
                
                                  
                
                loss=-model.LSM_likelihood_bias(epoch=epoch)/N
             
                
         
             
                optimizer.zero_grad() # clear the gradients.   
                loss.backward() # backpropagate
                optimizer.step() # update the weights
                writer.add_scalar("loss", loss.detach(), epoch)
                if epoch%1000==0:
                      print('Iteration Number:', epoch)
                      print('Negative Log-Likelihood:',(loss.item()*N)/elements)
                      if args.LP:
                          roc,pr=model.link_prediction() 
                          print('AUC-ROC:',roc)
                          print('AUC-PR:',pr)
            writer.close()


root = 'D:/study/thesis/project/HBDM-main/ppi_results/models/'+name+'/'
#os.makedirs(root)


def is_sparse(tensor):
    return isinstance(tensor, torch.Tensor) and tensor.layout == torch.sparse_coo
# Specify the variables to be saved
variables_to_save = {
    'k_exp_dist': [tensor.detach().cpu().numpy() if not is_sparse(tensor) else tensor.to_dense().detach().cpu().numpy() for tensor in model.k_exp_dist],
    'final_idx': [tensor.detach().cpu().numpy() for tensor in model.final_idx],
    'general_cl_id': [tensor.detach().cpu().numpy() for tensor in model.general_cl_id],
    'general_mask': [tensor.detach().cpu().numpy() for tensor in model.general_mask],
    'RE':[tensor.detach().cpu().numpy() for tensor in model.gamma],
    'latent':[tensor.detach().cpu().numpy() for tensor in model.latent_z]
}


# Loop through the variables and save them with their names as file names
for var_name, var_data in variables_to_save.items():
    file_path = os.path.join(root, var_name + '.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(var_data, f)

    

    
    
    
    
    
