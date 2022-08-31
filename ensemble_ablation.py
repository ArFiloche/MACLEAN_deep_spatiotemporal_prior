import numpy as np
import torch
import time

from architectures import*
from deepprior import*
from utils import*

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ',device)

### Param
T = 32 # window size
i = 0 # sample
n_z = 100 # latent space dim.
n_epoch = 2000 #optim. iteration
N = 30 # ensemble member

### Data
Truth = torch.Tensor(np.load('./data/ground_truth/ground_truth.npy'))[:,-1-128:-1,0:128]
Obs = torch.Tensor(np.load('./data/observations/ssh_tracks.npy'))[:,-1-128:-1,0:128]
Mask = (-1*torch.Tensor(np.load('./data/observations/mask.npy'))+1)[:,-1-128:-1,0:128]
Oi = torch.Tensor(np.load('./data/estimate/OI/oi_1year.npy'))[:,-1-128:-1,0:128]

### Sample
truth = Truth[i:i+T,:,:]
obs = Obs[i:i+T,:,:]
obs[obs!=obs]=2
mask = Mask[i:i+T,:,:]
oi = Oi[i:i+T,:,:]

results = torch.zeros((N,truth.shape[0],truth.shape[1],truth.shape[2]))

for n in range(N):

    ### Net / input
    net = DCGNet(n_z=100, n_channel=T)
    mu = torch.randn(n_z, 1, 1)
    #mu = torch.ones(n_z, 1, 1,1)
    deepprior=DeepPrior(net,n_epoch=n_epoch)

    deepprior.fit(inpt=mu, Obs=obs, Mask=mask, device=device)
    estimate = deepprior.X.detach()#.numpy()
    
    results[n,:,:,:] = estimate
    
    np.save('data/results/ensemble_ablation.npy', results)