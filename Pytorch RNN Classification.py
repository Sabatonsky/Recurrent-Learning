# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:11:45 2024

@author: Bannikov Maxim
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import os
from nltk import pos_tag, word_tokenize

device = torch.device('cpu') 
url = "https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/robert_frost.txt"
s=requests.get(url).text.rstrip().lower()

def get_tags(s): #Tokenize words in poems, then transform them in part of speech
    tuples = pos_tag(word_tokenize(s))
    return [y for x, y in tuples] #for each output of pos_tag func get only pos part

def get_poetry_classifier_data(samples_per_class, load_cached=True, save_cached=True):
    datafile = 'D:/Training_code/poetry_classifier_data.npz' #Save transformed texts into specific file.
    if load_cached and os.path.exists(datafile): #Load that file
        npz = np.load(datafile, allow_pickle=True) 
        X = npz['arr_0'] 
        Y = npz['arr_1']
        V = int(npz['arr_2'])
        return X, Y, V 
    #Else, create data from scratch and save it.
    word2idx = {}
    current_idx = 0
    X = []
    Y = []
    eap_dir = 'https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/edgar_allan_poe.txt'
    rf_dir = 'https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/robert_frost.txt'
    #Create inputs + labels for each text
    for fn, label in zip((eap_dir, rf_dir), (0, 1)):
        count = 0
        s=requests.get(fn).text.lower().split('\n')
        for line in s:
            line = line.rstrip()
            if line:
                tokens = get_tags(line)
                if len(tokens) > 1:
                    for token in tokens:
                        if token not in word2idx:
                            word2idx[token] = current_idx
                            current_idx += 1
                    sequence = np.array([word2idx[w] for w in tokens])
                    X.append(sequence)
                    Y.append(label)
                    count += 1
                    if count >= samples_per_class:
                        break
    X = np.array(X, dtype=object)
    if save_cached:
        np.savez(datafile, X, Y, current_idx, allow_pickle=True)
    return X, Y, current_idx #We are predicting by Part of speech, but for further analysis
#We don't actually need to know what part of speech it was. Therefore we just take current_idx that is our dimentionality D.

X, Y, current_idx = get_poetry_classifier_data(500, load_cached=True, save_cached=True)

class SimpleRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.i2h = nn.Linear(vocab_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, hidden_state):
        x = self.i2h(x)
        hidden_state = self.h2h(hidden_state)
        hidden_state = torch.tanh(x + hidden_state)
        return self.h2o(hidden_state), hidden_state
    
    def init_hidden(self):
        return torch.zeros(self.hidden_size, requires_grad=False)

lr = 10e-6
num_epochs = 2000
vocab_size = current_idx
hidden_size = 30
n_total_steps = len(X)
num_classes = 2

model = SimpleRNN(hidden_size, vocab_size, num_classes)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)

epoch_loss = []
acc_list = []

for epoch in range(num_epochs):
    losses = []
    predictions = np.zeros(Y.shape[0], dtype=int)
    for i, (x, y) in enumerate(zip(X, Y)):
        x = torch.tensor(x, dtype=torch.int64).reshape(1, -1)
        x = F.one_hot(x, vocab_size).float()
        y = torch.tensor(y, dtype=torch.int64).reshape(1)
        hidden = model.init_hidden()
        x, y, hidden = x.to(device), y.to(device), hidden.to(device)
        #Prepare x, y, hidden for torch.
        
        model.zero_grad()
        
        for word in range(x.shape[1]):
            output, hidden = model(x[:, word], hidden)
        
        loss = criterion(output, y)
        prediction = torch.max(output, 1).indices
        predictions[i] = prediction.item()
        
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    scheduler.step()       
    n_samples = Y.shape[0]
    n_correct = (predictions == Y).sum()
    train_acc = 100.0 * n_correct / n_samples
    acc_list.append(train_acc)
        
    epoch_loss.append(sum(losses)/n_total_steps)
    print(f'epoch {epoch+1} / {num_epochs}, loss = {epoch_loss[-1]:.4f}')
    print(f'accuracy = {acc_list[-1]:.2f}')
    
plt.plot(epoch_loss)
plt.show()

plt.plot(acc_list)
plt.show()