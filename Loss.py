import torch
import torch.nn as nn

def reconstruction_loss(x, y):
    return torch.sum((x - y) ** 2)

def kl_loss(x, x_var):
    return torch.sum(0.5*(x**2 - x_var**2 - 2*torch.log(x_var)))

class MainLoss(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(self, x, x_var, y):
        return -reconstruction_loss(x,y) + kl_loss(x,x_var)