import torch
import torch.nn as nn

class KLDLoss(nn.Module):
    ''' VAE Loss '''
    def __init__(self):
        super().__init__()
    
    def forward(self, mu, log_var):
        kld_loss = -0.5*torch.sum(1+log_var-mu**2-log_var.exp(), dim=1)
        return kld_loss.sum()