import numpy as np
import math

import torch

from torch.nn import _reduction as _Reduction
from torch.nn.functional import conv2d

def _fspecial_gaussian(size, channel, sigma):
    coords = torch.tensor([(x - (size - 1.) / 2.) for x in range(size)])
    coords = -coords ** 2 / (2. * sigma ** 2)
    grid = coords.view(1, -1) + coords.view(-1, 1)
    grid = grid.view(1, -1)
    grid = grid.softmax(-1)
    kernel = grid.view(1, 1, size, size)
    kernel = kernel.expand(channel, 1, size, size).contiguous()
    return kernel

def ssim(inpt, target, max_val=1, k1=0.01, k2=0.03, channel=1):
    
    inpt=inpt.unsqueeze(0).unsqueeze(0)
    target = target.unsqueeze(0).unsqueeze(0)
    kernel=_fspecial_gaussian(size=11, channel=1, sigma=1.5)
    
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    mu1 = conv2d(inpt, kernel, groups=channel)
    mu2 = conv2d(target, kernel, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(inpt * inpt, kernel, groups=channel) - mu1_sq
    sigma2_sq = conv2d(target * target, kernel, groups=channel) - mu2_sq
    sigma12 = conv2d(inpt * target, kernel, groups=channel) - mu1_mu2

    v1 = 2 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2

    ssim = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)
    
    return ssim.mean().item()

def SSIM(Oi,Truth):
    res = np.zeros(Oi.shape[0])
    for t in range(Oi.shape[0]):
        res[t]=ssim(torch.Tensor(Oi[t,:,:]),torch.Tensor(Truth[t,:,:]))
        
    return res
        

def SSIM_En(En,Truth):
    res = np.zeros((En.shape[0],En.shape[1]))
    for i in range(En.shape[0]):
        for t in range(En.shape[1]):
            res[i,t]=ssim(torch.Tensor(En[i,t,:,:]),torch.Tensor(Truth[t,:,:]))
        
    return res

def n_RMSE(estimate, truth):
    
    rmse = RMSE(estimate, truth)
    rms = RMSE(np.zeros(truth.shape), truth)
    n_rmse = np.ones(truth.shape[0]) - rmse/rms
    
    return n_rmse

def RMSE(estimate, truth):
    
    err = ((truth-estimate)**2).mean(axis=-1).mean(axis=-1)
    rmse = np.sqrt(err)
    
    return rmse

def rmse(x,ref):
    
    rmse=np.sqrt(((x-ref)**2).mean())
    
    return rmse