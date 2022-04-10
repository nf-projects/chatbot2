import json
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

#open the json file 'intents.json' in read mode
with open('intents.json') as f:
    # Each intent is one "category" - with tags, patterns (input), and responses
    intents = json.load(f)

# All of the words in the training data
all_words = []
# All tags in the training data
tags = []
# Holds patterns and tags
xy = []

# STEP 1: Tokenize each pattern
for intent in intents['intents']:
     tag = intent['tag']
     tags.append(tag)
     for pattern in intent['patterns']:
         # apply tokenization to each pattern
         w = tokenize(pattern)
         # using extend instead of append to avoid a 2D list :D
         all_words.extend(w)
         xy.append((w, tag))

# STEP 2: Remove punctuation and make everything lowercase
punctuation_characters = ['?', '!', '.', ',', ';', ':', '-', '_', '*', '@', '#', '$', '%', '^', '&', '+', '=', '~', '`', '{', '}', '|', '\\', '<', '>', '/', '"', "'"]
# STEP 3: Stem each word & remove punctuation characters
all_words = [stem(w) for w in all_words if w not in punctuation_characters]
# STEP 4: Sort and remove duplicates
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# STEP 5: Bag of words
x_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # Create bag of words comparing frequency from the pattern sentence to all words
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    
    label = tags.index(tag) # get the index of the tag
    y_train.append(label) # CrossEntropyLoss, whatever that means lol

# Create numpy arrays instead of Python arrays for performance reasons i guess
x_train = np.array(x_train)
y_train = np.array(y_train)

# A dataset required for the Pytorch stuff idk
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples

# Hyperparameters
batch_size = 8
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000


dataset = ChatDataset()
# This is the magic that makes it work i think (PyTorch)
# - dataset is the input dataset obviously
# - batch_size is the number of samples per batch to load
# - Shuffle=true makes it shuffle every epoch
# - num-workers=2 makes it multi-threaded
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# use the gpu if it's available, otherwise use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
# one of the things i have no idea what it does
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device, torch.int64)

        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

# Save the data generated

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
#save the data to a file
torch.save(data, FILE)

print(f'training complete! file saved to {FILE}')
