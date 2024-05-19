# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 20:40:32 2024

@author: Bannikov Maxim
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import string
import requests
import nltk
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import pickle
device = torch.device('cpu')

#!wget -nc https://lazyprogrammer.me/course_files/nlp/ner_train.pkl
#!wget -nc https://lazyprogrammer.me/course_files/nlp/ner_test.pkl

with open('ner_train.pkl', 'rb') as f:
  corpus_train = pickle.load(f)

with open('ner_test.pkl', 'rb') as f:
  corpus_test = pickle.load(f)

train_inputs = []
train_targets = []
#Split pickle to two lists of lists
for sentence_tag_pairs in corpus_train:
  tokens = []
  target = []
  for token, tag in sentence_tag_pairs:
    tokens.append(token)
    target.append(tag)
  train_inputs.append(tokens)
  train_targets.append(target)

test_inputs = []
test_targets = []
#Split pickle to two lists of lists
for sentence_tag_pairs in corpus_test:
  tokens = []
  target = []
  for token, tag in sentence_tag_pairs:
    tokens.append(token)
    target.append(tag)
  test_inputs.append(tokens)
  test_targets.append(target)

max_vocab_size = None
should_lowercase = False
word_tokenizer = Tokenizer(
    num_words = max_vocab_size,
    lower = should_lowercase,
    oov_token = 'UNK'
)

word_tokenizer.fit_on_texts(train_inputs)
train_inputs_int = word_tokenizer.texts_to_sequences(train_inputs)
test_inputs_int = word_tokenizer.texts_to_sequences(test_inputs)

word2idx = word_tokenizer.word_index
print('Found %s unique tokens.' %len(word2idx))

def flatten(list_of_lists):
  flattened = [val for sublist in list_of_lists for val in sublist]
  return flattened

all_train_targets = set(flatten(train_targets))
all_test_targets = set(flatten(test_targets))
print(all_train_targets==all_test_targets)
output_size = len(all_train_targets) + 1

tag_tokenizer = Tokenizer()
tag_tokenizer.fit_on_texts(train_targets)
train_targets_int = tag_tokenizer.texts_to_sequences(train_targets)
test_targets_int = tag_tokenizer.texts_to_sequences(test_targets)

def collate_fn(batch):
    x, y = zip(*batch)
    x_pad = pad_sequence(x, batch_first=True)
    y_pad = pad_sequence(y, batch_first=True)
    return x_pad, y_pad

class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = [torch.LongTensor(sublist) for sublist in inputs]
        self.labels = [torch.LongTensor(sublist) for sublist in labels]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        input = self.inputs[index]
        label = self.labels[index]
        return input, label

class Custom_CrossEntropyLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.cel = nn.CrossEntropyLoss()

  def forward(self, yhat, y):
    mask = torch.where(y == 0, 0, 1)
    mask_ = mask.unsqueeze(-2).expand(output.size())
    masked_yhat = mask_ * yhat
    masked_y = mask * y
    loss = self.cel(masked_yhat, masked_y)
    return loss

batch_size = 100
train_dataset = CustomDataset(train_inputs_int, train_targets_int)
test_dataset = CustomDataset(test_inputs_int, test_targets_int)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
train_loader_no_pad = torch.utils.data.DataLoader(dataset=train_dataset)
test_loader_no_pad = torch.utils.data.DataLoader(dataset=test_dataset)

class Custom_BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super().__init__()
        self.embedding_dim = embedding_dim #We size
        self.vocab_size = vocab_size #V size
        self.hidden_size = hidden_size #M size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        #final output layer
        self.h2y = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        x = self.embeddings(x)
        out, _ = self.lstm(x)
        return self.h2y(out)

if __name__ == '__main__':
  #hyperparams
  lr = 10e-4 #To be tuned
  num_epochs = 20 #To be tuned
  vocab_size = len(word2idx) + 1
  embedding_dim = 32 #To be tuned
  hidden_size = 32 #To be tuned

  n_total_steps = len(train_loader)
  model = Custom_BiLSTM(vocab_size, embedding_dim, hidden_size, output_size)

  #loss and optimizer
  criterion = Custom_CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  epoch_loss = []
  epoch_acc = []

  #training loop
  for epoch in range(num_epochs):
    losses = []
    acc_list = []
    for x, y in train_loader:
      x, y = x.to(device), y.to(device)
      model.zero_grad()

      #forward
      output = model(x)
      output = torch.transpose(output, 1, 2)
      loss = criterion(output, y)
      predictions = torch.max(output, 1).indices

      #backwards
      loss.backward()
      optimizer.step()

      losses.append(loss.item())
      mask = torch.where(y == 0, 0, 1)
      n_samples = mask.sum()
      n_correct = ((predictions == y) * mask).sum().item()
      train_acc = 100.0 * n_correct / n_samples
      acc_list.append(train_acc)

    epoch_loss.append(sum(losses)/n_total_steps)
    epoch_acc.append(sum(acc_list)/n_total_steps)
    print(f'epoch {epoch+1} / {num_epochs}, loss = {epoch_loss[-1]:.4f}')
    print(f'accuracy = {epoch_acc[-1]:.2f}')

  plt.plot(epoch_loss)
  plt.savefig('epoch_loss')
  plt.clf()

  plt.plot(epoch_acc)
  plt.savefig('epoch_acc')
  plt.clf()

def metrics(name, data_loader):
  with torch.no_grad():
    targets = []
    predictions = []
    for x, y in data_loader:
      x, y = x.to(device), y.to(device)
      output = model(x)
      mask = torch.where(y == 0, 0, 1)
      prediction = torch.max(output, 2).indices * mask
      predictions.extend(prediction.flatten().tolist())
      targets.extend(y.flatten().tolist())
  acc = accuracy_score(targets, predictions)
  f1 = f1_score(targets, predictions, average='macro')
  print(f'{name} acc: {acc:.4f}')
  print(f'{name} f1 score: {f1:.4f}')

metrics("Train", train_loader_no_pad)
metrics("Test", test_loader_no_pad)
