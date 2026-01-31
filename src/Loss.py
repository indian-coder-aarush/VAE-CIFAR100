import torch
import torch.nn as nn
import torch.nn.functional as F

def reconstruction_loss(x, y):
    return torch.mean(((x- y) ** 2))

def kl_loss(z, z_var):
    return -0.5 * torch.mean(1 + z_var*0.1 - z.pow(2) - (z_var*0.1).exp())

class MainLoss(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, y, z , z_var, beta):
        return reconstruction_loss(x,y) + beta*kl_loss(z,z_var)

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
            self.current_step = self.cycle
            self.plus = False
        if self.current_step <= -self.cycle:
            self.current_step = -self.cycle
            self.plus = True
        return current_beta*self.max_beta