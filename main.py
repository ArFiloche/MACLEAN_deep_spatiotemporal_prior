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
n_z = 100 # latent space dim.
n_epoch = 2000 #optim. iteration

### Data
Truth = torch.Tensor(np.load('./data/ground_truth/ground_truth.npy'))[:,-1-128:-1,0:128]
Obs = torch.Tensor(np.load('./data/observations/ssh_tracks.npy'))[:,-1-128:-1,0:128]
Mask = (-1*torch.Tensor(np.load('./data/observations/mask.npy'))+1)[:,-1-128:-1,0:128]
Oi = torch.Tensor(np.load('./data/estimate/OI/oi_1year.npy'))[:,-1-128:-1,0:128]

results = torch.zeros((T,Truth.shape[0],Truth.shape[1],Truth.shape[2]))

for i in range(Truth.shape[0]-T+1):
    
    if i%10==0:
        print('sample',i)
    ### Sample
    truth=Truth[i:i+T,:,:]
    obs = Obs[i:i+T,:,:]
    obs[obs!=obs]=2
    mask = Mask[i:i+T,:,:]

    ### Net / input
    net = ST_DCGNet(n_z=n_z, n_channel=T)
    #mu = torch.randn(n_z, 1, 1,1)
    mu = torch.ones(n_z, 1, 1,1)
    deepprior=DeepPrior(net,n_epoch=n_epoch)

    deepprior.fit(inpt=mu, Obs=obs, Mask=mask, device=device)
    estimate = deepprior.X.detach()#.numpy()
    
    for t in range(T):
        results[t,t+i,:,:] = estimate[t,:,:]
    
    np.save('data/results/main.npy', results)