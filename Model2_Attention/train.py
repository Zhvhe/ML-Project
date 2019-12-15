#!/usr/bin/env python
# coding: utf-8

# Credits to https://towardsdatascience.com/generate-piano-instrumental-music-by-using-deep-learning-80ac35cdbd2e
# 


#Import required packages
#!pip install tensorflow==2.0.0-alpha0
#!pip install pretty_midi

import tensorflow as tf
from tensorflow.keras import backend as K
import glob
import random
import pretty_midi
import IPython
import numpy as np
from tqdm import tnrange, tqdm_notebook, tqdm
from random import shuffle, seed
import numpy as np
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD
import numpy as np
from numpy.random import choice
import pickle
import matplotlib.pyplot as plt

import unicodedata
import re
import numpy as np
import os
import io
import time
import pre_processing
import model



# Get Sample 100 midi files from the dataset for training the network
sampled_100_midi = get_all_midi[0:100]  

#Assign sequence length
seq_len = 100

# Create a map of note -> index using NoteEncoder 
# Transform these list of nodes into input / output format to Neural Network
NoteEncoderbatch = 1
start_index = 0
note_encoder = pre_processing.NoteEncoder()

for i in tqdm_notebook(range(len(sampled_100_midi))):
    dict_time_notes = pre_processing.create_dict_time_notes(sampled_100_midi, batch_song=1, start_index=i, use_tqdm=False, fs=5)
    full_notes = pre_processing.process_notes_in_song(dict_time_notes)
    for note in full_notes:
        note_encoder.partial_fit(list(note.values()))

#Add an empty note for generating music from a note encoding of sequence length 100
note_encoder.add_new_note('e')


unique_notes = note_encoder.unique_word
print(unique_notes)

model = model.create_model(seq_len, unique_notes)

model.summary()


class TrainModel:
  
  def __init__(self, epochs, note_encoder, sampled_100_midi, frame_per_second, 
               batch_nnet_size, batch_song, optimizer, checkpoint, loss_fn,
               checkpoint_prefix, total_songs, model):
    self.epochs = epochs
    self.note_encoder = note_encoder
    self.sampled_100_midi = sampled_100_midi
    self.frame_per_second = frame_per_second
    self.batch_nnet_size = batch_nnet_size
    self.batch_song = batch_song
    self.optimizer = optimizer
    self.checkpoint = checkpoint
    self.loss_fn = loss_fn
    self.checkpoint_prefix = checkpoint_prefix
    self.total_songs = total_songs
    self.model = model
    
  def train(self):
    for epoch in tqdm_notebook(range(self.epochs),desc='epochs'):
      # shufle the midi files for each iteration
      shuffle(self.sampled_100_midi)
      loss_total = 0
      steps = 0
      steps_nnet = 0

      # Iterate until song list size
      for i in tqdm_notebook(range(0,self.total_songs, self.batch_song), desc='MUSIC'):

        steps += 1
        inputs_nnet_large, outputs_nnet_large = create_midi_batches(
            self.sampled_100_midi, self.batch_song, start_index=i, fs=self.frame_per_second, 
            seq_len=seq_len, use_tqdm=False) 
        inputs_nnet_large = np.array(self.note_encoder.transform(inputs_nnet_large), dtype=np.int32)
        outputs_nnet_large = np.array(self.note_encoder.transform(outputs_nnet_large), dtype=np.int32)

        index_shuffled = np.arange(start=0, stop=len(inputs_nnet_large))
        np.random.shuffle(index_shuffled)

        for nnet_steps in tqdm_notebook(range(0,len(index_shuffled),self.batch_nnet_size)):
          steps_nnet += 1
          current_index = index_shuffled[nnet_steps:nnet_steps+self.batch_nnet_size]
          inputs_nnet, outputs_nnet = inputs_nnet_large[current_index], outputs_nnet_large[current_index]
          
          # Handles exception if BATCH_SONG is not compatible with Batch_NNET_SIZE during the last batch
          if len(inputs_nnet) // self.batch_nnet_size != 1:
            break
          loss = self.train_step(inputs_nnet, outputs_nnet)
          loss_total += tf.math.reduce_sum(loss)
          if steps_nnet % 20 == 0:
            print("epochs {} | Steps {} | total loss : {}".format(epoch + 1, steps_nnet, loss_total))

      checkpoint.save(file_prefix = self.checkpoint_prefix)
  
  @tf.function
  def train_step(self, inputs, targets):
    with tf.GradientTape() as tape:
      prediction = self.model(inputs)
      loss = self.loss_fn(targets, prediction)
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    return loss


# Train the network
EPOCHS = 50
BATCH_SONG = 8
BATCH_NNET_SIZE = 96
TOTAL_SONGS = len(sampled_100_midi)
FRAME_PER_SECOND = 5

import os

# Optimization using below list from Tensorflow to reduce the loss and make the model converge to effective value
def create_optimizer(optimizer_type):
	return optimizer_type()
	
""" optimizer_Adadelta = Adadelta()
optimizer_Adagrad = Adagrad()
optimizer_Adam = Adam()
optimizer_Adamax = Adamax()
optimizer_Ftrl = Ftrl()
optimizer_Nadam = Nadam()
optimizer_RMSprop = RMSprop()
optimizer_SGD = SGD() """

#optimizer_list = [optimizer_Adadelta, optimizer_Adagrad, optimizer_Adam, optimizer_Adamax, optimizer_Ftrl, optimizer_Nadam, optimizer_RMSprop, optimizer_SGD]
optimizer_list = [Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD]
#optimizer_list = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD']

#Triggered with different epochs and settled with 50 epochs as it took ~68 hours to run the entire optimizer list for 50 epochs

#Loop over the optimizers and plot the losses
for opt in optimizer_list:
	print("Optimizer: {}".format(opt))
	checkpoint = tf.train.Checkpoint(optimizer=create_optimizer(opt),
									 model=model)
	checkpoint_dir = './training_checkpoints'
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	loss_fn = sparse_categorical_crossentropy

	train_class = TrainModel(EPOCHS, note_encoder, sampled_100_midi, FRAME_PER_SECOND,
					  BATCH_NNET_SIZE, BATCH_SONG, create_optimizer(opt), checkpoint, loss_fn,
					  checkpoint_prefix, TOTAL_SONGS, model)

	train_class.train()
	# Save the model and The Tokenizer generated from sample
	model.save(str(opt) + 'model_ep50.h5')
	pickle.dump( note_encoder, open( (str(opt) + "tokenizer50.p"), "wb" ) )


