#!/usr/bin/env python
# coding: utf-8

# In[4]:


import nltk
nltk.download('punkt')


# In[21]:


from nltk.stem.porter import PorterStemmer
stemmer= PorterStemmer ()
import numpy as np

def tokenize (sentence):
    
    return nltk.word_tokenize(sentence)

def stem (word):
    return stemmer. stem(word.lower())

def bag_of_words (tokenized_sentence, all_words):
   """
   sentence= ["hello", "hi", "how", "are", "you"]
   words= ["hello", "hi", "I", "bye", "you", "thanks", "cool"]
   bag  = [   1,     1,    0,   0,     1,     0,      0]
   """ 
   tokenized_sentence = [stem(w) for w in tokenized_sentence]
    
   bag = np.zeros(len(all_words),dtype = np.float32)
   for idx, w in enumerate (all_words):
       if w in tokenized_sentence: 
            bag[idx]= 1.0
            
   return bag


# In[ ]:




