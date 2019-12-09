#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

import parsing_midi
import models


# In[2]:


#number of training runs through model 
epochs = 200
#size of batch of inputs per epoch
epoch_size = 64
#path to put the weight files
weightpath = "olivia_run_0"
#length of input sequence of notes to generate the next note
sequence_length = 100


# In[3]:



#This section creates the set of output notes
def prepare_sequence_out(notes, n_vocab):
    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_output = []

    # create the outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_out = notes[i + sequence_length]
        network_output.append(note_to_int[sequence_out])

    network_output = np_utils.to_categorical(network_output)

    return network_output


# In[4]:


#This section trains the model
def train(model, network_input, network_output):
    #save checkpoints every epoch run
    filepath = weightpath+"weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=epochs, batch_size=epoch_size, callbacks=callbacks_list)


# In[5]:


#Main function to cover grabbing the data to training the model
#sets up parameters to train the network and then runs
def train_network():
    notes = parsing_midi.get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))

    #get inputs and outputs
    network_input, normalized_input = parsing_midi.prepare_sequence_in(notes, n_vocab, sequence_length)
    network_output = prepare_sequence_out(notes, n_vocab)

    model = models.create_lstm_network_with_batch(normalized_input, n_vocab)

    train(model, normalized_input, network_output)


# In[6]:


#if called from commandline, this should just try training the network
#if called in a notebook, will just run
if __name__ == '__main__':
    train_network()


# In[ ]:





# In[ ]:




