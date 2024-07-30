# -*- coding: utf-8 -*-
"""
@author: Bannikov Maxim

"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#https://jaketae.github.io/study/pytorch-rnn/

#Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyper parameters
num_epochs = 200
batch_size = 100
lr = 10e-6

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

class Custom_RNN(nn.Module):
  def __init__(self, input_size, output_size, hidden_size, batch_size):
    super().__init__()
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.batch_size = batch_size

    self.rnn = nn.RNN(input_size, hidden_size, 1, batch_first=True)
    self.i2h = nn.Linear(input_size, hidden_size, bias = False)
    self.h2h = nn.Linear(hidden_size, hidden_size)
    self.h2o = nn.Linear(hidden_size, output_size)

  def forward(self, x, hidden_state):
    x = self.i2h(x)
    hidden_state = self.h2h(hidden_state)
    hidden_state = torch.tanh(x + hidden_state)
    return self.h2o(hidden_state), hidden_state

  def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size, requires_grad=False)

X, Y = all_parity_pairs(12)
data = MyDataset(X, Y)
num_features = X.shape[1]
num_classes = len(set(Y))
num_hidden = 4

X.shape

train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)

model = Custom_RNN(num_features, num_classes, num_hidden, batch_size)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

n_total_steps = len(train_loader)
losses = []
acc_list = []

for epoch in range(num_epochs):
  for i, (x, y) in enumerate(train_loader):

    if x.shape[0] != model.batch_size:
      continue

    hidden = model.init_hidden()
    x, y, hidden = x.to(device), y.to(device), hidden.to(device)

    model.zero_grad()

    #forward
    output, hidden = model(x, hidden)
    loss = criterion(output, y)

    #backwards
    loss.backward()
    optimizer.step()

    if (i+1) % 10 == 0:
      losses.append(loss.item())
      print(f'epoch {epoch+1} / {num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')
      _, predictions = torch.max(output, 1)
      n_samples = y.shape[0]
      n_correct = (predictions == y).sum().item()
      train_acc = 100.0 * n_correct / n_samples
      acc_list.append(train_acc)
      print(f'accuracy = {train_acc}')

plt.plot(losses)
plt.savefig('losses')
plt.clf()
