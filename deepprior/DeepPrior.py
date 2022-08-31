import torch
import torch.optim as optim
import torch.nn as nn

class DeepPrior():
    
    def __init__(self, net,
                 lr=0.01, beta1=0.9, n_epoch=2000):
    
        self.net=net
        
        self.n_epoch=n_epoch
        self.lr=lr
        self.beta1=beta1
        
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.lr, betas=(self.beta1, 0.999))

        self.X=0
        self.losses=[]

    
    def J(self, X, Obs, Mask): 
        
        j = ((Obs - X)*Mask*(Obs - X)).mean()  
        
        return j
    

    def fit(self, inpt, Obs, Mask, device='cpu'):
        
        device=torch.device(device)
        
        self.net.to(device)
        inpt = inpt.to(device)
        Obs = Obs.to(device)
        Mask = Mask.to(device)
        
        for i in range(self.n_epoch):
            
            self.net.zero_grad()
            loss=0
        
            X=self.net.forward(inpt).squeeze(0)
            self.X=X
            
            loss = self.J(X, Obs, Mask)
                
            self.losses.append(loss.item())
            
            loss.backward(retain_graph=True)
            self.optimizer.step()