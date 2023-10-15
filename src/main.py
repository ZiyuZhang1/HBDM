
import argparse
import torch
import numpy as np
import torch.optim as optim
import sys
from tqdm import tqdm
import pickle

print(sys.path)

sys.path.append('./src/')
print(sys.path)
parser = argparse.ArgumentParser(description='Hierarchical Block Distance Model')
#####for now changed 
parser.add_argument('--W', type=eval, 
                      choices=[True, False], default=True,
                    help='activates random effects')

parser.add_argument('--RE', type=eval, 
                      choices=[True, False], default=True,
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
                    help='performs link prediction'

                    
                    )

parser.add_argument('--D', type=int, default=9, metavar='N',
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
    writer = SummaryWriter(log_dir=r"D:\study\thesis\project\HBDM-main\ppi_results\trained curve\{}".format(name))
    
    latent_dims=[args.D]
    datasets=[args.dataset]
    for dataset in datasets:
        for latent_dim in latent_dims:
            if args.W:
                sparse_i = []
                sparse_j = []
                for level in range(1,10):
                    inputfile = "./datasets/"+dataset+'/level_'+str(level)
                    sparse_i_level=torch.from_numpy(np.loadtxt(inputfile+'_sparse_i.txt')).long().to(device)
                    # input data, link column positions with i<j
                    sparse_j_level=torch.from_numpy(np.loadtxt(inputfile+'_sparse_j.txt')).long().to(device)
                    sparse_i.append(sparse_i_level)
                    sparse_j.append(sparse_j_level)
                # with open('test_i', 'wb') as file:
                #     pickle.dump(sparse_i, file)
                # with open('test_j', 'wb') as file:
                #     pickle.dump(sparse_j, file)
            else:
                # input data, link rows i positions with i<j
                sparse_i=torch.from_numpy(np.loadtxt("./datasets/"+dataset+'/sparse_i.txt')).long().to(device)
                # input data, link column positions with i<j
                sparse_j=torch.from_numpy(np.loadtxt("./datasets/"+dataset+'/sparse_j.txt')).long().to(device)
            
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
                N=int(torch.cat(sparse_j).max()+1)
            else: 
                N=int(sparse_j.max()+1)
            # Missing data here denoted if Marginalization is applied or not
            # In case missing data is set to True then the input should be the complete graph
            model = LSM(torch.randn(N,latent_dim),sparse_i,sparse_j,N,latent_dim=latent_dim,non_sparse_i=non_sparse_i,non_sparse_j=non_sparse_j,sparse_i_rem=sparse_i_rem,sparse_j_rem=sparse_j_rem,CVflag=True,graph_type='undirected',missing_data=False,device=device,LP=args.LP).to(device)
            optimizer = optim.Adam(model.parameters(), args.lr)  
            elements=(N*(N-1))*0.5
            for epoch in tqdm(range(args.epochs),desc="HBDM is Running…",ascii=False, ncols=75):
                if (epoch%args.RH==0):
                    model.build_hierarchy=True
                
                loss=-model.LSM_likelihood_bias(epoch=epoch)/N

                optimizer.zero_grad() # clear the gradients.   
                loss.backward() # backpropagate
                optimizer.step() # update the weights
                writer.add_scalar("Negative Log-Likelihood", (loss.item()*N)/elements, epoch)
                if epoch%1000==0:
                      print('Iteration Number:', epoch)
                      print('Negative Log-Likelihood:',(loss.item()*N)/elements)
                      if args.LP:
                          roc,pr=model.link_prediction() 
                          print('AUC-ROC:',roc)
                          print('AUC-PR:',pr)
            writer.close()
    latent_representations = model.latent_z
    
    root = 'D:/study/thesis/project/HBDM-main/ppi_results'
    latent_path = root+'/latent/'+name+'.pkl'
    model_path = root+'/models/'+name+'.pkl'
    # Serialize and save the Tensor to the file
    with open(latent_path, 'wb') as file:
        pickle.dump(latent_representations, file)
    # Close the file
    file.close()
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    

    
    
    
    
    
