from Data.DataLoader import make_dataloader
from Loss import MainLoss
from Layers import Model
from Data import DataLoader
import torch

def train_model(model, optimizer, loss_fn, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        print("Epoch " + str(epoch))
        for batch_idx, data in enumerate(train_loader):

            generated, z, variance  = model(data)
            loss = loss_fn(generated, data, z, variance)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print("     At Batch " + str(batch_idx) + " Loss: " + str(loss.item()))

model = Model()
Optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
loss_fn = MainLoss()
train, test = DataLoader.load_all_data()
train_loader = make_dataloader(train)
val_loader = make_dataloader(test)
epochs = 20

train_model(model, Optimizer, loss_fn, train_loader, val_loader, epochs)