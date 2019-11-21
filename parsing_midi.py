#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord


# In[ ]:


#path to input training data, songs
midi_path = "midi_songs"
#path to notes notefile
notes_path = "notes_olivia_quarter_length"


# In[ ]:


#This section is for reading in the midi file into python
def get_notes():
    notes = []

    for file in glob.glob(midi_path+"/*.mid"):
        midi = converter.parse(file)

        #print("Parsing %s" % file)

        notes_to_parse = None

        #check if file has instrument parts
        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            #check to see if this is a note or chord
            if isinstance(element, note.Note):
                notes.append(str(element.pitch) + "-" + str(element.quarterLength))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder) + "-" + str(element.quarterLength))

    with open("data/"+notes_path, 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


# In[ ]:


#This section creates the set of input notes sequences
def prepare_sequence_in(notes, n_vocab, sequence_length):

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)


# In[ ]:


#This section is for reading in the midi file into python
def get_notes_for_one_song(song):
    song_notes = []
    midi = converter.parse(midi_path+'/'+song+".mid")
    notes_to_parse = None

    #check if file has instrument parts
    try: # file has instrument parts
        s2 = instrument.partitionByInstrument(midi)
        notes_to_parse = s2.parts[0].recurse() 
    except: # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    offset_m = 0.0
    offest_c = 0.0
    for element in notes_to_parse:
        #check to see if this is a note or chord
        if isinstance(element, note.Note):
            song_notes.append(str(element.pitch) + "-" + str(element.quarterLength))
            #print(str(element.pitch) + "-" + str(element.quarterLength))
        elif isinstance(element, chord.Chord):
            song_notes.append('.'.join(str(n) for n in element.normalOrder) + "-" + str(element.quarterLength))
            #print(str('.'.join(str(n) for n in element.normalOrder)) + "-" + str(element.quarterLength))
            
    return song_notes


# In[ ]:


#This section creates the set of input notes sequences
def prepare_melody_in(notes, n_vocab, song):

    song_notes = get_notes_for_one_song(song)
    
    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    
    input_song = [note_to_int[char] for char in song_notes]

    return input_song


# In[ ]:


#et_notes()


# In[ ]:




