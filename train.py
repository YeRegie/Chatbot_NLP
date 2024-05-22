#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from nltk_utils import tokenize



# In[2]:


with open('intents.json','r') as f:
    intents = json.load(f)
    
print (intents)


# In[3]:


from nltk_utils import stem

all_words = []
tags =[]
xy =[]

for intent in intents['intents']:
    tag = intent ['tag']
    tags.append(tag)
    for pattern in intent ['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))
        
ignore_words = ['?','!','.',',']
all_words = [stem (w) for w in all_words if w not in ignore_words ]
print ( all_words)


# In[4]:


all_words =  sorted(set(all_words))
tags =sorted(set(tags))

print(tags)


# In[21]:


import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader




# In[22]:


from nltk_utils import bag_of_words

X_train = []
y_train =[]

for (pattern_sentence,tag) in xy:
    bag= bag_of_words (pattern_sentence, all_words)
    X_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label)
    
X_train = np.array (X_train)
y_train = np.array (y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)





# In[26]:

from model import NeuralNet

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet (input_size, hidden_size, output_size).to(device)
    
    #loss and optimizer
    
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
            
            #backward and optimizer step
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (epoch +1) %100 == 0:
            print (f'epoch {epoch+1}/{num_epochs},loss= {loss.item():.4f}')
            
print (f'final loss, loss= {loss.item():.4f}')


data = {
    "model_state": model.state_dict(),
    'input_size': input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words":all_words,
    "tags": tags
    
}

FILE = "data.pth"
torch.save(data,FILE)

print(f'training complete. file savedd to {FILE}')
        
    #dataset[index]
    def _getitem_(self,index):
        return self.xdata[index],self.ydata[index]
    def _len_(self):
        return self.n_samples
    
    #hyperparameters
    batch_size = 8
    hidden_size = 8
    output_size = len(tags)
    input_size = len(X_train[0])
    
    print(input_size, len(all_words))
    print(output_size,tags)
    
    dataset =ChatDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle = true , num_workers = 2)
    
    model = nlp( input_size, hidden_size,output_size)
        
   


# In[ ]:




