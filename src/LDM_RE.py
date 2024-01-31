import torch
import torch.nn as nn
import numpy as np
import torch_sparse
from fractal_main_cond import Tree_kmeans_recursion
from spectral_clustering_G import Spectral_clustering_init
from sklearn import metrics
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

# device = torch.device("cpu")
# torch.set_default_tensor_type('torch.FloatTensor')
undirected=1





class LSM(nn.Module,Tree_kmeans_recursion,Spectral_clustering_init):
    def __init__(self,data,sparse_i,sparse_j, sparse_w, input_size,latent_dim,graph_type,non_sparse_i=None,non_sparse_j=None,sparse_i_rem=None,sparse_j_rem=None,CVflag=False,initialization=None,scaling=None,missing_data=False,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),LP=True):
        super(LSM, self).__init__()
        Tree_kmeans_recursion.__init__(self,minimum_points=3*int(data.shape[0]/(data.shape[0]/np.log(data.shape[0]))),init_layer_split=3*torch.round(torch.log(torch.tensor(data.shape[0]).float())))
        Spectral_clustering_init.__init__(self,device=device)
        self.input_size=input_size
        self.cluster_evolution=[]
        self.mask_evolution=[]
        self.init_layer_split=torch.round(torch.log(torch.tensor(data.shape[0]).float()))
        self.init_layer_idx=torch.triu_indices(int(self.init_layer_split),int(self.init_layer_split),1)
       
        self.bias=nn.Parameter(torch.randn(1,device=device))
        self.scaling_factor=nn.Parameter(torch.randn(1,device=device))
        self.latent_dim=latent_dim
        self.initialization=1
        self.gamma=nn.Parameter(torch.randn(self.input_size,device=device))
        self.build_hierarchy=False
        self.graph_type=graph_type
        self.scaling=1
        #create indices to index properly the receiver and senders variable
        self.sparse_i_idx=sparse_i
        self.flag1=0
        self.sparse_j_idx=sparse_j
        self.sparse_w=sparse_w
        self.pdist = nn.PairwiseDistance(p=2,eps=0)
        self.missing_data=missing_data
        self.CUDA=True
        self.pdist_tol1=nn.PairwiseDistance(p=2,eps=0)

        self.device=device
        
        if LP:
            self.non_sparse_i_idx_removed=non_sparse_i
         
            self.non_sparse_j_idx_removed=non_sparse_j
               
            self.sparse_i_idx_removed=sparse_i_rem
            self.sparse_j_idx_removed=sparse_j_rem
            self.removed_i=torch.cat((self.non_sparse_i_idx_removed,self.sparse_i_idx_removed))
            self.removed_j=torch.cat((self.non_sparse_j_idx_removed,self.sparse_j_idx_removed))

        
        self.spectral_data=self.spectral_clustering()#.flip(1)

        self.first_centers_sp=torch.randn(int(self.init_layer_split),self.spectral_data.shape[1],device=device)

        global_cl,spectral_leaf_centers=self.kmeans_tree_z_initialization(depth=80,initial_cntrs=self.first_centers_sp) 
           
        self.first_centers=torch.randn(int(torch.round(torch.log(torch.tensor(data.shape[0]).float()))),latent_dim,device=device)
      

        spectral_centroids_to_z=spectral_leaf_centers[global_cl]
        # spectral_centroids_to_z=self.spectral_data
        if self.spectral_data.shape[1]>latent_dim:
            self.latent_z=nn.Parameter(spectral_centroids_to_z[:,0:latent_dim])
        elif self.spectral_data.shape[1]==latent_dim:
            self.latent_z=nn.Parameter(spectral_centroids_to_z)
        else:
            self.latent_z=nn.Parameter(torch.zeros(self.input_size,latent_dim,device=device))
            self.latent_z.data[:,0:self.spectral_data.shape[1]]=spectral_centroids_to_z
        # self.latent_z=nn.Parameter(torch.zeros(self.input_size,latent_dim,device=device))

    #introduce the likelihood function containing the two extra biases gamma_i and alpha_j
    def square_loss(self,epoch):
        '''

        Parameters
        ----------
        cent_dist : real
            distnces of the updated centroid and the k-1 other centers.
        count_prod : TYPE
            DESCRIPTION.
        mask : Boolean
            DESCRIBES the slice of the mask for the specific kmeans centroid.

        Returns
        -------
        None.

        '''
        self.epoch=epoch
        
            
        z_pdist=(((self.latent_z[self.sparse_i_idx]-self.latent_z[self.sparse_j_idx]+1e-06)**2).sum(-1))**0.5
        # y_pre = -z_pdist+self.gamma[self.sparse_i_idx]+self.gamma[self.sparse_j_idx]+self.bias
        y_pre = -z_pdist+self.gamma[self.sparse_i_idx]+self.gamma[self.sparse_j_idx]
        square_loss_sparse = torch.sum((y_pre-self.sparse_w)**2)
        
    #############################################################################################################################################################        
                
        
        return square_loss_sparse
    
    def link_prediction(self):
        with torch.no_grad():
            z_pdist_miss=(((self.latent_z[self.removed_i]-self.latent_z[self.removed_j])**2).sum(-1))**0.5
            logit_u_miss=-z_pdist_miss+self.gamma[self.removed_i]+self.gamma[self.removed_j]
            rates=torch.exp(logit_u_miss)

            target=torch.cat((torch.zeros(self.non_sparse_i_idx_removed.shape[0]),torch.ones(self.sparse_i_idx_removed.shape[0])))
            precision, recall, thresholds = metrics.precision_recall_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())

           
        return metrics.roc_auc_score(target.cpu().data.numpy(),rates.cpu().data.numpy()),metrics.auc(recall,precision)
    
    