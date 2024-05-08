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



# In[26]:


from model import nlp


class ChatDataset(Dataset):
    def _init_(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
        
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




