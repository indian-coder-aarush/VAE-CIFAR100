from Data.DataLoader import make_dataloader
from src.Loss import MainLoss
from src.Layers import Model
from Data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt

def train_model(model, optimizer, loss_fn, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        print("Epoch " + str(epoch))
        loss_sum = 0
        for batch_idx, data in enumerate(train_loader):
            generated, z, variance  = model(data)
            loss = loss_fn(generated, data, z, variance)
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()
            optimizer.zero_grad()

            if(epoch == 30 and batch_idx == 130):
                plt.imshow((generated[0].detach().numpy().transpose(1,2,0)))
                plt.show()
                plt.imshow(data[0].detach().numpy().transpose(1, 2, 0))
                plt.show()

            print('Overall loss after Epoch ' + str(epoch) + ' is ' + str(loss_sum/395.125))
            print("     At Batch " + str(batch_idx) + " Loss: " + str(loss.item()))

model = Model()
Optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
loss_fn = MainLoss()
train, test = DataLoader.load_all_data()
train_loader = make_dataloader(train)
val_loader = make_dataloader(test)
epochs = 20000

train_model(model, Optimizer, loss_fn, train_loader, val_loader, epochs)