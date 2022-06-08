# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 10:49:03 2022

@author: nnak
"""

import argparse
import torch
import numpy as np
import torch.optim as optim
import sys
from tqdm import tqdm
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

sys.path.append('./src/')

from HM_LDM import LSM


parser = argparse.ArgumentParser(description='Hybrid Membership-Latent Distance Model')

parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs for training (default: 10K)')

parser.add_argument('--scaling_epochs', type=int, default=2000, metavar='N',
                    help='number of epochs for learning initial scale for the random effects (default: 2K)')

parser.add_argument('--delta_sq', type=float, default=10, metavar='N',
                    help='delta^2 hyperparameter controlling the volume of the simplex')

parser.add_argument('--cuda', type=eval, 
                      choices=[True, False],  default=True,
                    help='CUDA training')

parser.add_argument('--p', type=int, 
                      choices=[1, 2],  default=1,
                    help='L2 norm power (default: 1)')


parser.add_argument('--LP',type=eval, 
                      choices=[True, False], default=True,
                    help='performs link prediction')

parser.add_argument('--D', type=int, default=8, metavar='N',
                    help='dimensionality of the embeddings (default: 8)')

parser.add_argument('--lr', type=float, default=0.1, metavar='N',
                    help='learning rate for the ADAM optimizer, for large values of delta 0.01 is more stable (default: 0.1)')

parser.add_argument('--sample_percentage', type=float, default=1, metavar='N',
                    help='Sample size network percentage, it should be equal or less than 1 (default: 1)')



parser.add_argument('--dataset', type=str, default='grqc',
                    help='dataset to apply HM-LDM on')



args = parser.parse_args()

if  args.cuda:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')


    
    


if __name__ == "__main__":
    latent_dims=[args.D]
    datasets=[args.dataset]
    for dataset in datasets:
        for latent_dim in latent_dims:
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
                
            N=int(sparse_j.max()+1)
            #Missing data here denoted if Marginalization is applied or not
            # In case missing data is set to True then the input should be the complete graph
            sample_size=int(args.sample_percentage*N)
            model = LSM(sparse_i,sparse_j,N,latent_dim=latent_dim,sample_size=sample_size,non_sparse_i=non_sparse_i,non_sparse_j=non_sparse_j,sparse_i_rem=sparse_i_rem,sparse_j_rem=sparse_j_rem,CVflag=True,graph_type='undirected',missing_data=False,device=device,p=args.p).to(device)
            optimizer = optim.Adam(model.parameters(), args.lr)  
            if args.p==1:
                model.reg_l=args.delta_sq**0.5
            else:
                model.reg_l=args.delta_sq
            elements=(N*(N-1))*0.5
            for epoch in tqdm(range(args.epochs),desc="HM-LDM is Running…",ascii=False, ncols=75):
                if epoch==args.scaling_epochs:
                    model.scaling=0
                
                                  
                
                loss=-model.LSM_likelihood_bias(epoch=epoch)/sample_size
             
                
         
             
                optimizer.zero_grad() # clear the gradients.   
                loss.backward() # backpropagate
                optimizer.step() # update the weights
                if epoch%1000==0:
                      print('Iteration Number:', epoch)
                      print('Negative Log-Likelihood:',(loss.item()*N)/elements)
                      if args.LP:
                          roc,pr=model.link_prediction() 
                          print('AUC-ROC:',roc)
                          print('AUC-PR:',pr)
                          
    plt.rcParams["figure.figsize"] = (10,10)
    
    z_idx=model.latent_z.argmax(1)
    w_idx=model.latent_z.argmax(1)
    
    f_z=z_idx.argsort()
    f_w=w_idx.argsort()
    
    new_i=torch.cat((sparse_i,sparse_j))
    new_j=torch.cat((sparse_j,sparse_i))
    
    D=csr_matrix((np.ones(new_i.shape[0]),(new_i.cpu().numpy(),new_j.cpu().numpy())),shape=(N,N))#.todense()
 
    
    D = D[:, f_w.cpu().numpy()][f_z.cpu().numpy()]
    
    
    plt.spy(D,markersize=1)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'adjacency_{args.dataset}.pdf')
    plt.show()
    
    

                          
                          
                          
                          
                          
                          
                          
                         
