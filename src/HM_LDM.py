
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:05:57 2021

@author: nnak
"""

# Import all the packages
# Import all the packages
import torch
import torch.nn as nn
from spectral_clustering import Spectral_clustering_init
from sklearn import metrics
#from blobs import *
# import sparse 

from torch_sparse import spspmm




class LSM(nn.Module,Spectral_clustering_init):
    def __init__(self,sparse_i,sparse_j, input_size,latent_dim,graph_type,sample_size,non_sparse_i=None,non_sparse_j=None,sparse_i_rem=None,sparse_j_rem=None,CVflag=False,initialization=None,scaling=None,missing_data=False,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),p=1):
        super(LSM, self).__init__()
        # initialization
        # initialization
        Spectral_clustering_init.__init__(self,num_of_eig=latent_dim,method='Normalized_sym')
        self.input_size=input_size
     
       
        self.bias=nn.Parameter(torch.randn(1,device=device))
        self.scaling_factor=nn.Parameter(torch.randn(1,device=device))
        self.latent_dim=latent_dim
        
        self.gamma=nn.Parameter(torch.randn(self.input_size,device=device))
        
        #self.alpha=nn.Parameter(torch.randn(self.input_size,device=device))
        self.graph_type=graph_type
        self.initialization=1
        self.scaling=1
        #create indices to index properly the receiver and senders variable
        self.sparse_i_idx=sparse_i
        self.flag1=0
        self.sparse_j_idx=sparse_j
        self.pdist = nn.PairwiseDistance(p=2,eps=0)
        self.missing_data=missing_data
        
        self.sampling_weights=torch.ones(self.input_size,device=device)
        self.sample_size=sample_size

        self.p=p
       
        self.non_sparse_i_idx_removed=non_sparse_i
     
        self.non_sparse_j_idx_removed=non_sparse_j
           
        self.sparse_i_idx_removed=sparse_i_rem
        self.sparse_j_idx_removed=sparse_j_rem
        if self.non_sparse_i_idx_removed!=None:
            # total sample of missing dyads with i<j
            self.removed_i=torch.cat((self.non_sparse_i_idx_removed,self.sparse_i_idx_removed))
            self.removed_j=torch.cat((self.non_sparse_j_idx_removed,self.sparse_j_idx_removed))

        self.Softmax=nn.Softmax(1)

        self.spectral_data=self.spectral_clustering()

       
      

        spectral_centroids_to_z=self.spectral_data
        if self.spectral_data.shape[1]>latent_dim:
            self.latent_z1=nn.Parameter(spectral_centroids_to_z[:,0:latent_dim])
        elif self.spectral_data.shape[1]==latent_dim:
            self.latent_z1=nn.Parameter(spectral_centroids_to_z)
        else:
            self.latent_z1=nn.Parameter(torch.zeros(self.input_size,latent_dim,device=device))
            self.latent_z1.data[:,0:self.spectral_data.shape[1]]=spectral_centroids_to_z
    
            

    

    
    def sample_network(self):
        # USE torch_sparse lib i.e. : from torch_sparse import spspmm

        # sample for undirected network
        sample_idx=torch.multinomial(self.sampling_weights, self.sample_size,replacement=False)
        # translate sampled indices w.r.t. to the full matrix, it is just a diagonal matrix
        indices_translator=torch.cat([sample_idx.unsqueeze(0),sample_idx.unsqueeze(0)],0)
        # adjacency matrix in edges format
        edges=torch.cat([self.sparse_i_idx.unsqueeze(0),self.sparse_j_idx.unsqueeze(0)],0)
        # matrix multiplication B = Adjacency x Indices translator
        # see spspmm function, it give a multiplication between two matrices
        # indexC is the indices where we have non-zero values and valueC the actual values (in this case ones)
        indexC, valueC = spspmm(edges,torch.ones(edges.shape[1]), indices_translator,torch.ones(indices_translator.shape[1]),self.input_size,self.input_size,self.input_size,coalesced=True)
        # second matrix multiplication C = Indices translator x B, indexC returns where we have edges inside the sample
        indexC, valueC=spspmm(indices_translator,torch.ones(indices_translator.shape[1]),indexC,valueC,self.input_size,self.input_size,self.input_size,coalesced=True)
        
        # edge row position
        sparse_i_sample=indexC[0,:]
        # edge column position
        sparse_j_sample=indexC[1,:]
     
        
        if self.missing_data: 
            # missing
            missing_edges=torch.cat([self.sparse_i_idx_removed.unsqueeze(0),self.sparse_j_idx_removed.unsqueeze(0)],0)
            # matrix multiplication B = Adjacency x Indices translator
            # see spspmm function, it give a multiplication between two matrices
            # indexC is the indices where we have non-zero values and valueC the actual values (in this case ones)
            indexC, valueC = spspmm(missing_edges,torch.ones(missing_edges.shape[1]), indices_translator,torch.ones(indices_translator.shape[1]),self.input_size,self.input_size,self.input_size,coalesced=True)
            # second matrix multiplication C = Indices translator x B, indexC returns where we have edges inside the sample
            indexC, valueC=spspmm(indices_translator,torch.ones(indices_translator.shape[1]),indexC,valueC,self.input_size,self.input_size,self.input_size,coalesced=True)
            
            # edge row position
            sparse_i_sample_removed=indexC[0,:]
            # edge column position
            sparse_j_sample_removed=indexC[1,:]
            
 
        
            
            missing_positions=torch.cat([self.removed_i.unsqueeze(0),self.removed_j.unsqueeze(0)],0)
            # matrix multiplication B = Adjacency x Indices translator
            # see spspmm function, it give a multiplication between two matrices
            # indexC is the indices where we have non-zero values and valueC the actual values (in this case ones)
            indexC, valueC = spspmm(missing_positions,torch.ones(missing_positions.shape[1]), indices_translator,torch.ones(indices_translator.shape[1]),self.input_size,self.input_size,self.input_size,coalesced=True)
            # second matrix multiplication C = Indices translator x B, indexC returns where we have edges inside the sample
            indexC, valueC=spspmm(indices_translator,torch.ones(indices_translator.shape[1]),indexC,valueC,self.input_size,self.input_size,self.input_size,coalesced=True)
            
            # edge row position
            non_sparse_i_sample_removed=indexC[0,:]

            # edge column position
            non_sparse_j_sample_removed=indexC[1,:]
           

            return sample_idx,sparse_i_sample,sparse_j_sample,sparse_i_sample_removed,sparse_j_sample_removed,non_sparse_i_sample_removed,non_sparse_j_sample_removed
        else:
            return sample_idx,sparse_i_sample,sparse_j_sample
        
    
    
        
    #introduce the likelihood function 
    def LSM_likelihood_bias(self,epoch):
        '''
        Poisson log-likelihood ignoring the log(k!) constant
        
        '''
        self.latent_z=self.Softmax(self.latent_z1)
        

        if self.scaling:
            sample_idx,sparse_sample_i,sparse_sample_j=self.sample_network()
            mat=torch.exp(torch.zeros(sample_idx.shape[0],sample_idx.shape[0])+1e-06)

            z_pdist1=0.5*torch.mm(torch.exp(self.gamma[sample_idx].unsqueeze(0)),(torch.mm((mat-torch.diag(torch.diagonal(mat))),torch.exp(self.gamma[sample_idx]).unsqueeze(-1))))
            z_pdist2=(self.gamma[sparse_sample_i]+self.gamma[sparse_sample_j]).sum()
    
           

    
            log_likelihood_sparse=z_pdist2-z_pdist1
            
           
        else:
            sample_idx,sparse_sample_i,sparse_sample_j=self.sample_network()
            if self.p==2:
                mat=torch.exp(-self.reg_l*((torch.cdist(self.latent_z[sample_idx],self.latent_z[sample_idx],p=2)+1e-04)**2)+1e-06)
            else:
                mat=torch.exp(-self.reg_l*((torch.cdist(self.latent_z[sample_idx],self.latent_z[sample_idx],p=2))))

            z_pdist1=0.5*torch.mm(torch.exp(self.gamma[sample_idx].unsqueeze(0)),(torch.mm((mat-torch.diag(torch.diagonal(mat))),torch.exp(self.gamma[sample_idx]).unsqueeze(-1))))
            if self.p==2:
                z_pdist2=(-self.reg_l*((((self.latent_z[sparse_sample_i]-self.latent_z[sparse_sample_j]+1e-04)**2).sum(-1)))+self.gamma[sparse_sample_i]+self.gamma[sparse_sample_j]).sum()
            else:
                z_pdist2=(-self.reg_l*((((self.latent_z[sparse_sample_i]-self.latent_z[sparse_sample_j]+1e-06)**2).sum(-1)))**0.5+self.gamma[sparse_sample_i]+self.gamma[sparse_sample_j]).sum()

           

    
            log_likelihood_sparse=z_pdist2-z_pdist1
        
        
        return log_likelihood_sparse
    
    
    
    
   
    
    def link_prediction(self):
        self.latent_z=self.Softmax(self.latent_z1)

        with torch.no_grad():
            if self.p==2:
                z_pdist_miss=(((self.latent_z[self.removed_i]-self.latent_z[self.removed_j]+1e-06)**2).sum(-1))
            else:
                z_pdist_miss=(((self.latent_z[self.removed_i]-self.latent_z[self.removed_j])**2).sum(-1))**0.5
            if self.scaling:
                logit_u_miss=self.gamma[self.removed_i]+self.gamma[self.removed_j]

            else:
                logit_u_miss=-self.reg_l*z_pdist_miss+self.gamma[self.removed_i]+self.gamma[self.removed_j]
            rates=logit_u_miss
            self.rates=rates

            target=torch.cat((torch.zeros(self.non_sparse_i_idx_removed.shape[0]),torch.ones(self.sparse_i_idx_removed.shape[0])))
            #fpr, tpr, thresholds = metrics.roc_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())
            precision, tpr, thresholds = metrics.precision_recall_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())

           
        return metrics.roc_auc_score(target.cpu().data.numpy(),rates.cpu().data.numpy()),metrics.auc(tpr,precision)
    
    
    
    
 
        
         
