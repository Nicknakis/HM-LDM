# Hybrid-Membership Latent Distance Model (HM-LDM)

Python 3.8.3 and Pytorch 1.9.0 implementation of the Hybrid-Membership Latent Distance Model (HM-LDM).

## Description
A central aim of modeling complex networks is to accurately embed networks in order to detect structures and predict link and node properties. The latent space models (LSM) have become prominent frameworks for embedding networks and include the latent distance (LDM) and eigenmodel (LEM) as the most widely used LSM specifications. For latent community detection, the embedding space in LDMs has been endowed with a clustering model whereas LEMs have been constrained to part-based non-negative matrix factorization (NMF) inspired representations promoting community discovery. We presently reconcile LSMs with latent community detection by constraining the LDM representation to the D-simplex forming the hybrid-membership latent distance model (HM-LDM). We show that for sufficiently large simplex volumes this can be achieved without loss of expressive power whereas by extending the model to squared Euclidean distances, we recover the LEM formulation with constraints promoting part-based representations akin to NMF. Importantly, by systematically reducing the volume of the simplex, the model becomes unique and ultimately leads to hard assignments of nodes to simplex corners. We demonstrate experimentally how the proposed HM-LDM admits accurate node representations in regimes ensuring identifiability and valid community extraction. Importantly, HM-LDM naturally reconciles soft and hard community detection with network embeddings exploring a simple continuous optimization procedure on a volume constrained simplex that admits the systematic investigation of trade-offs between hard and mixed membership community detection.

### Unipartite network example based on the inferred HM-LDM memberships for [AstroPh](http://snap.stanford.edu/data/ca-AstroPh.html) and [Facebook](http://snap.stanford.edu/data/ego-Facebook.html)  Networks 

| <img src="https://github.com/Nicknakis/Hybrib-Membership-Latent-Distance-Model/blob/main/images/astroph.jpg?raw=true"  alt="drawing"  width="150"  />   | <img src="https://github.com/Nicknakis/Hybrib-Membership-Latent-Distance-Model/blob/main/images/astroph_l2.jpg?raw=true"  alt="drawing"  width="150" />  | <img src="https://github.com/Nicknakis/Hybrib-Membership-Latent-Distance-Model/blob/main/images/facebook.jpg?raw=true"  alt="drawing"  width="150"  />  | <img src="https://github.com/Nicknakis/Hybrib-Membership-Latent-Distance-Model/blob/main/images/facebook_l2.jpg?raw=true"  alt="drawing"  width="150"  />  |
|:---:|:---:|:---:|:---:|
| AstroPh (p=2) | AstroPh (p=1)| Facebook (p=2) | Facebook (p=1) |


### A Bipartite Example with a [Drug-Gene](http://snap.stanford.edu/biodata/datasets/10002/10002-ChG-Miner.html) Network

| <img src="https://github.com/Nicknakis/Hybrib-Membership-Latent-Distance-Model/blob/main/images/drug_gene_1.jpeg?raw=true"  alt="drawing"  width="220"  />   | <img src="https://github.com/Nicknakis/Hybrib-Membership-Latent-Distance-Model/blob/main/images/l2_drug_gene_1.jpeg?raw=true"  alt="drawing"  width="220"  />  |
|:---:|:---:|
| Drug-Gene (p=2) | Drug-Gene (p=1) |

### Installation
pip install -r requirements.txt

Our Pytorch implementation uses the [pytorch_sparse](https://github.com/rusty1s/pytorch_sparse) package. Installation guidelines can be found at the corresponding [Github repository](https://github.com/rusty1s/pytorch_sparse).

### Learning identifiable graph representations with HM-LDM
**RUN:** &emsp; python main.py

optional arguments:

**--epochs**  &emsp;  number of epochs for training (default: 10K)

**--scaling_epochs**    &emsp;    number of epochs for learning initial scale for the random effects (default: 2K)

**--cuda**  &emsp;    CUDA training (default: True)

**--LP**   &emsp;     performs link prediction (default: True)

**--D**   &emsp;      dimensionality of the embeddings (default: 8)

**--lr**   &emsp;     learning rate for the ADAM optimizer (default: 0.1)

**--p**   &emsp;     L2 norm power (default: 1)

**--dataset** &emsp;  dataset to apply HM-LDM (default: grqc)

**--sample_percentage** &emsp;  sample size network percentage, it should be less than 1 (default: 1)

**--delta_sq** &emsp;  delta^2 hyperparameter controlling the volume of the simplex


### CUDA Implementation

The code has been primarily constructed and optimized for running in a GPU-enabled environment.


### References
N. Nakis, A. Celikkanat, and M. Mørup, [Hybrib-Membership-Latent-Distance-Model](https://arxiv.org/pdf/2206.03463.pdf), Preprint.

