#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.layers import BatchNormalization as BatchNorm

# In[5]:


#LSTM configurations settings

#percentage of inputs dropped put pf model to prevent overfitting
lstm_dropout = 0.3
#number of LSTM nodes in the layer
lstm_nodes = 512
#the type of activation method we are using
lstm_activation_method='softmax'
#size of internal layers
lstm_dense = 256
#method of loss calculation
lstm_loss='categorical_crossentropy'
#method of optimization
lstm_optimizer='rmsprop'


# In[6]:


#This creates a LSTM RNN
def create_lstm_network(network_input, n_vocab):
    #This is a sequential model
    model = Sequential()
    #Add a LSM layer that has a certain number of nodes, 
    #knows the shape of the input data,
    #and is outputting something sequential
    model.add(LSTM(
        lstm_nodes,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=lstm_dropout,
        return_sequences=True
    ))
    #This deliberatly looses part of the data to prevent overfitting
    model.add(LSTM(lstm_nodes, return_sequences=True, recurrent_dropout=lstm_dropout))
    model.add(LSTM(lstm_nodes))
    model.add(BatchNorm())
    model.add(Dropout(lstm_dropout))
    model.add(Dense(lstm_dense))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(lstm_dropout))
    #This makes sure our output layer matches the number of possible outputs
    model.add(Dense(n_vocab))
    #activation method set here
    model.add(Activation(lstm_activation_method))
    model.compile(loss=lstm_loss, optimizer=lstm_optimizer)

    return model


# In[ ]:




