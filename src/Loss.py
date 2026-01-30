import torch
import torch.nn as nn

def reconstruction_loss(x, y):
    return torch.mean(((x- y) ** 2))

def kl_loss(z, z_var):
    return -0.5 * torch.mean(1 + z_var*0.1 - z.pow(2) - (z_var*0.1).exp())

class MainLoss(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, y, z , z_var):
        return reconstruction_loss(x,y) + 0.5*kl_loss(z,z_var)