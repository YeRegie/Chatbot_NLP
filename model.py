#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
import torch.nn as nn


# In[4]:


class nlp(nn.Module):
    def _init_(self, input_size, hidden_size, num_classes):
        super (nlp, self)._init_()
        self.l1 =nn.linear(input_size, hidden_size)
        self.l2 =nn.linear(hidden_size, hidden_size)
        self.l3 =nn.linear(input_size, num_classes)
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        #no activtion and no softmax
        return out


# In[ ]:




