import json
from my_nltk import tokenize, stem, bag_of_words
from torch.utils.data import Dataset, DataLoader
from model import NN
import torch
import torch.nn as nn
import torch.optim as optim

with open('intents.json', 'r') as f:
    dic = json.load(f)

tags = []
all_words = []
xy = []
for intent in dic['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['.', '/', '!', ',', '?']
all_words = [stem(i) for i in all_words if i not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []
for (pattern_words, tag) in xy:
    bag = bag_of_words(pattern_words, all_words)  # imp
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)
    
class ChatDataset():
    def __init__(self):
        self.nsamples = len(X_train)
        self.xdata = X_train
        self.ydata = y_train
        
    def __getitem__(self, index):
        return self.xdata[index], self.ydata[index]

    def __len__(self):
        return self.nsamples

# Hyperparameters
batch_size = 8
ip_size = len(X_train[0])
hidden_size = 8
op_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NN(ip_size, hidden_size, op_size).to(device)

# Loss & Optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for data in train_loader:
        X, y = data
        X = X.to(device)
        y = y.to(device)

        y_ = model(X)
        loss = loss_function(y_, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

data = {
    'model_state':model.state_dict(),
    'input_size':ip_size,
    'output_size':op_size,
    'hidden_size':hidden_size,
    'all_words':all_words,
    'tags':tags
}

FILE = 'data.pth'
torch.save(data, FILE)








        