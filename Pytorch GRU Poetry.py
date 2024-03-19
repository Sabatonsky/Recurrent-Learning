# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 14:46:29 2024

@author: Bannikov Maxim
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import string
import requests

device = torch.device('cpu') 
url = "https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/robert_frost.txt"
s=requests.get(url).text.rstrip().lower()

def remove_puctuation(s):
    return s.translate(str.maketrans('', '', string.punctuation))

def get_robert_frost():
    word2idx = {'START': 0, 'END': 1}
    current_idx = 2
    sentences = []
    for line in s.split('\n'):
        print(line)
        line = line.strip()
        if line:
            tokens = remove_puctuation(line.lower()).split()
            sentence = []
            for t in tokens:
                if t not in word2idx:
                    word2idx[t] = current_idx
                    current_idx += 1
                idx = word2idx[t]
                sentence.append(idx)
            sentences.append(sentence)
    return sentences, word2idx

sentences, word2idx = get_robert_frost()

class Custom_RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding_dim = embedding_dim #We size
        self.vocab_size = vocab_size #V size
        self.hidden_size = hidden_size #M size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.i2h = nn.Linear(embedding_dim, hidden_size, bias = False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.i2z = nn.Linear(embedding_dim, hidden_size, bias = False)
        self.h2z = nn.Linear(hidden_size, hidden_size)
        self.i2r = nn.Linear(embedding_dim, hidden_size, bias = False)
        self.h2r = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden_state):
        x = self.embeddings(x)
        x_gate = self.i2r(x)
        x_rate = self.i2z(x)
        x = self.i2h(x)
        
        hidden_state_gate = self.h2r(hidden_state)
        gate = torch.sigmoid(x_gate + hidden_state_gate)
        #Gate is calculated before hidden_state_hat, and transfered to it.
        
        hidden_state_rate = self.h2z(hidden_state)
        rate = torch.sigmoid(x_rate + hidden_state_rate)
        
        hidden_state_hat = self.h2h(gate * hidden_state)
        hidden_state_hat = torch.tanh(x + hidden_state_hat) 

        hidden_state = (1 - rate) * hidden_state + rate * hidden_state_hat
        return self.h2o(hidden_state), hidden_state
    
    def init_hidden(self):
        return torch.zeros(self.hidden_size, requires_grad=False)
        #As N = 1, due to T of different length, we make hidden size (M)

    def generate(self, word2idx): #pi - probabilities of initial vector we use them to generate first word.
        idx2word = {v:k for k, v in word2idx.items()} #собираем обратный словарь для перевода.
        V = len(word2idx) #Считаем стартовый словарь.
        
        n_lines = 0 #Текущее количество стихов в куплете.
        
        X = [0] #Набиваем первое слово в стих.
        hidden = self.init_hidden()
        X = torch.tensor(X, dtype=torch.int64).reshape(1, -1)
        
        while n_lines < 4: #Пока нет 4 стихов в куплете.
            with torch.no_grad():
                X, hidden = X.to(device), hidden.to(device)
                output, hidden = self.forward(X[:,-1], hidden) #Предсказываем следующее слово по RNN.
                output = F.softmax(output, dim = 1)
                prediction = np.random.choice(V, p = output.numpy().flatten())
                prediction = torch.tensor(prediction, dtype=torch.int64).reshape(1, -1).to(device)
                X = torch.cat((X, prediction), 1) #Добавляем предсказанное слово в конец стиха.
                #В результате работы RNN. Check if true.
                if prediction.item() > 1:
                    word = idx2word[prediction.item()]
                    print(word, end = ' ')
                elif prediction.item() == 1: #Если это слово "END", значит нам необходимо перейти на следующую строчку
                    n_lines += 1
                    X = [0]
                    X = torch.tensor(X, dtype=torch.int64).reshape(1, -1)
                    print('')
                    hidden = model.init_hidden()

def generate_poetry():
    model = Custom_RNN(vocab_size, embedding_dim, hidden_size)
    path = "D:/Training_code/simpleRNN.pth"
    model.load_state_dict(torch.load(path))

    # determine initial state distribution for starting sentences
    model.generate(word2idx)

if __name__ == '__main__':
    #hyperparams
    lr = 10e-4 #To be tuned
    num_epochs = 200 #To be tuned
    vocab_size = len(word2idx)
    embedding_dim = 30 #To be tuned
    hidden_size = 30 #To be tuned
    
    n_total_steps = len(sentences)
    model = Custom_RNN(vocab_size, embedding_dim, hidden_size)
    
    #loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    epoch_loss = []
    epoch_acc = []
    
    #training loop
    for epoch in range(num_epochs):
        losses = []
        acc_list = []
        for x in sentences: #В связи с форматом данных мы можем использовать только SGD.
        #Так как временные ряды у нас разной длины. Чтобы ускорить процесс можно использовать либо бакетинг либо паддинг.
            if np.random.random() < 0.2:
                X = [0] + x #X is started with 0, as it is START token.
                Y = x + [1] #Y for 0 we predict first word from x and so on.
            else:
                X = [0] + x[:-1]
                Y = x
            #For last word we predict 1, that is END token.
            X = torch.tensor(X, dtype=torch.int64).reshape(1, -1)
            Y = torch.tensor(Y, dtype=torch.int64).reshape(1, -1)
            hidden = model.init_hidden()
            X, Y, hidden = X.to(device), Y.to(device), hidden.to(device)
            #Prepare X, Y, hidden for torch.
            
            model.zero_grad()
            loss = 0
            predictions = torch.zeros((X.shape[0], X.shape[1]))
            
            #forward
            for word in range(X.shape[1]):
                output, hidden = model(X[:,word], hidden)
                loss += criterion(output, Y[:, word])
                prediction = torch.max(output, 1).indices
                predictions[:, word] = prediction
            #Loop through T. Hidden is saved through one time series but is reinitiated in new one.
            #For each step t in X we take one word and pass it to RNN. Calculate loss for Y in t.
            #Save hidden for t+1
            
            #backwards
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            n_samples = Y.shape[0]*Y.shape[1]
            n_correct = (predictions == Y).sum().item()
            train_acc = 100.0 * n_correct / n_samples
            acc_list.append(train_acc)
    
        epoch_loss.append(sum(losses)/n_total_steps)
        epoch_acc.append(sum(acc_list)/n_total_steps)
        print(f'epoch {epoch+1} / {num_epochs}, loss = {epoch_loss[-1]:.4f}')
        print(f'accuracy = {epoch_acc[-1]:.2f}')
        
    plt.plot(epoch_loss)
    plt.show()
    
    plt.plot(epoch_acc)
    plt.show()
    
    path = "D:/Training_code/simpleRNN.pth"
    torch.save(model.state_dict(), path)
    
    generate_poetry()
