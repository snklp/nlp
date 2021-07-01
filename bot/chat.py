import  random
import json
import torch
from model import NN
from my_nltk import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = 'data.pth'
data = torch.load(FILE)

model_state = data['model_state']
input_size = data['input_size']
output_size = data['output_size']
hidden_size = data['hidden_size']
all_words = data['all_words']
tags = data['tags']

model = NN(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = 'Rakesh Tikait'
print("Aao Baaten Karen! type 'quit' to exit")

while True:
    sentence = input('You: ')
    if sentence == 'quit':break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    y_ = model(X)
    _, prdeicted = torch.max(y_, dim=1)
    tag = tags[prdeicted.item()]

    probs = torch.softmax(y_, dim=1)
    prob = probs[0][prdeicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:print('Rakesh Tikait: ', random.choice(intent['responces']))
    
    else:print('Rakesh_Tikait: Sorry Bhaisaab samjh nhi aao!!')