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

#Assign Sequence Length
seq_len = 100

#Optimizer list
optimizer_list = [Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD]

# Create a map of note -> index using NoteEncoder 
# Transform these list of nodes into input / output format to Neural Network
NoteEncoderbatch = 1
start_index = 0
note_encoder = NoteEncoder()

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

# Generate Music Midi Files

# Create random 100 notes and input to network
def generate_from_random(unique_notes, seq_len=100):
  generate = np.random.randint(0,unique_notes,seq_len).tolist()
  return generate
    
# 99 empty notes followed by a note of our choice.
def generate_from_single_note(note_encoder, new_notes='35'):
  generate = [note_encoder.notes_to_index['e'] for i in range(49)]
  generate += [note_encoder.notes_to_index[new_notes]]
  return generate

def generate_notes(generate, model, unique_notes, max_generated=1000, seq_len=100):
  for i in tqdm_notebook(range(max_generated), desc='genrt'):
    test_input = np.array([generate])[:,i:i+seq_len]
    predicted_note = model.predict(test_input)
    random_note_pred = choice(unique_notes+1, 1, replace=False, p=predicted_note[0])
    generate.append(random_note_pred[0])
  return generate

# Convert generated Piano roll array into MIDI format using pretty_midi
def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # Add 1 column padding to identify start and end of midi
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # track on/off of notes using their velocity
    velocity_changes = np.nonzero(np.diff(piano_roll).T)
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm



def create_midi_file(generate, midi_file_name = "result.mid", start_index=49, fs=8, max_generated=1000):
  note_string = [note_encoder.index_to_notes[ind_note] for ind_note in generate]
  array_piano_roll = np.zeros((128,max_generated+1), dtype=np.int16)
  for index, note in enumerate(note_string[start_index:]):
    if note == 'e':
      pass
    else:
      splitted_note = note.split(',')
      for j in splitted_note:
        array_piano_roll[int(j),index] = 1
  generate_to_midi = piano_roll_to_pretty_midi(array_piano_roll, fs=fs)
  print("Tempo {}".format(generate_to_midi.estimate_tempo()))
  for note in generate_to_midi.instruments[0].notes:
    note.velocity = 100
  generate_to_midi.write(midi_file_name)

# Load the model and the Tokenizer and generate midi files from network

max_generate = 200
unique_notes = note_encoder.unique_word
seq_len=100
for opt in optimizer_list:
	model = tf.keras.models.load_model((str(opt) + 'model_ep50.h5'), custom_objects=SeqSelfAttention.get_custom_objects())
	note_encoder  = pickle.load( open( (str(opt)+"tokenizer50.p"), "rb" ) )
	generate = generate_from_random(unique_notes, seq_len)
	generate = generate_notes(generate, model, unique_notes, max_generate, seq_len)
	create_midi_file(generate, str(opt)+"random.mid", start_index=seq_len-1, fs=7, max_generated = max_generate)


max_generate = 300
unique_notes = note_encoder.unique_word
seq_len=100
for opt in optimizer_list:
	model = tf.keras.models.load_model((str(opt) + 'model_ep50.h5'), custom_objects=SeqSelfAttention.get_custom_objects())
	note_encoder  = pickle.load( open( (str(opt)+"tokenizer50.p"), "rb" ) )
	generate = generate_from_single_note(note_encoder, '72')
	generate = generate_notes(generate, model, unique_notes, max_generate, seq_len)
	create_midi_file(generate, str(opt)+"single_note.mid", start_index=seq_len-1, fs=8, max_generated = max_generate)

