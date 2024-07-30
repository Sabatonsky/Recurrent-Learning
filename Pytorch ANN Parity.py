# -*- coding: utf-8 -*-
"""
@author: Bannikov Maxim

"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

#Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyper parameters
num_epochs = 100
batch_size = 20
lr = 10e-4

def all_parity_pairs(nbit):
  N=2**nbit
  remainder = 100 - (N % 100)
  Ntotal = N + remainder
  X = np.zeros((Ntotal, nbit))
  Y = np.zeros(Ntotal)
  for ii in range(Ntotal):
    i = ii % N
    for j in range(nbit):
      if i % (2**(j+1)) != 0:
        i -= 2**j
        X[ii, j] = 1
    Y[ii] = X[ii].sum() % 2
  return X, Y

class MyDataset(Dataset):
  def __init__(self, X, Y):
    self.x=torch.tensor(X, dtype=torch.float32)
    self.y=torch.tensor(Y, dtype=torch.int64)

  def __len__(self):
    return len(self.y)

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]

class NeuralNet(nn.Module):
  def __init__(self, input_size, output_size, hidden_sizes):
    super(NeuralNet, self).__init__()
    self.hidden = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
    self.hidden.extend([nn.Linear(hidden_sizes[i-1], hidden_sizes[i]) for i in range(1, len(hidden_sizes))])
    self.hidden.append(nn.Linear(hidden_sizes[-1], output_size))
    self.num_hidden = len(self.hidden)

  def forward(self, x):
    for i in range(self.num_hidden - 1):
      x = F.relu(self.hidden[i](x))
    x = self.hidden[-1](x)
    return x

X, Y = all_parity_pairs(12)
data = MyDataset(X, Y)
num_features = X.shape[1]
num_classes = len(set(Y))
num_hidden = [2048]
num_hidden = [1024]*2

train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)

model = NeuralNet(num_features, num_classes, num_hidden)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

n_total_steps = len(train_loader)
losses = []

summary(model, (200, num_features))

for epoch in range(num_epochs):
  for i, (x, y) in enumerate(train_loader):

    x = x.to(device)
    y = y.to(device)

    #forward
    outputs = model(x)
    loss = criterion(outputs, y)

    #backwards
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 10 == 0:
      losses.append(loss.item())

  if epoch % 10 == 0:
    print(f'epoch {epoch+1} / {num_epochs}, loss = {loss.item():.4f}')

plt.plot(losses)
plt.savefig('losses')
plt.clf()
