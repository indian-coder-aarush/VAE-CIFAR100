import torch
import torch.nn as nn
import torch.nn.functional as F

def reconstruction_loss(x, y):
    w = torch.tensor([1.5, 1.5, 1.5]).reshape(1,3,1,1)
    return torch.mean(torch.sum(w * torch.abs(x - y), dim=(1, 2, 3)))

def kl_loss(z_mean, z_var):
    kl = -0.5 * (1 + z_var - z_mean.pow(2) - z_var.exp())
    kl = torch.clamp(kl, min=0.05)
    kl = torch.mean(torch.sum(kl,(1,2,3)))
    return kl

class MainLoss(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, y, mean, z , z_var, beta):
        return reconstruction_loss(x,y) + beta*kl_loss(mean,z_var)

class KLBetaAnnealing:

    def __init__(self, cycle = 20, max_beta = 0.5):
        super().__init__()
        self.cycle = int(cycle/2)
        self.current_step = torch.tensor(-self.cycle)
        self.max_beta = max_beta
        self.plus = True

    def step(self):
        current_beta = F.sigmoid(self.current_step)
        if self.plus:
            self.current_step += 1
        else:
            self.current_step -= 1
        if self.current_step >= self.cycle:
            self.current_step = torch.tensor(self.cycle)
            self.plus = False
        if self.current_step <= -self.cycle:
            self.current_step = torch.tensor(-self.cycle)
            self.plus = True
        return current_beta*self.max_beta