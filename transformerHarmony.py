# # Environment Set-Up
# print('Copying checkpoints and Salamander piano SoundFont (via https://sites.google.com/site/soundfonts4u) from GCS...')
# !gsutil -q -m cp -r gs://magentadata/models/music_transformer/* /content/
# !gsutil -q -m cp gs://magentadata/soundfonts/Yamaha-C5-Salamander-JNv5.1.sf2 /content/
#
# print('Installing dependencies...')
# !apt-get update -qq && apt-get install -qq libfluidsynth1 build-essential libasound2-dev libjack-dev
# !pip install -qU magenta pyfluidsynth
#
# import ctypes.util
# def proxy_find_library(lib):
#   if lib == 'fluidsynth':
#     return 'libfluidsynth.so.1'
#   else:
#     return ctypes.util.find_library(lib)
# ctypes.util.find_library = proxy_find_library

print('Importing libraries...')

import numpy as np
import os
import tensorflow as tf

from google.colab import files

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib

import magenta.music as mm
from magenta.models.score2perf import score2perf

print('Done!')

# Defining constants and helper functions
SF2_PATH = '/content/Yamaha-C5-Salamander-JNv5.1.sf2'
SAMPLE_RATE = 16000

# Upload a MIDI file and convert to NoteSequence.
def upload_midi():
  data = list(files.upload().values())
  if len(data) > 1:
    print('Multiple files uploaded; using only one.')
  return mm.midi_to_note_sequence(data[0])

# Decode a list of IDs.
def decode(ids, encoder):
  ids = list(ids)
  if text_encoder.EOS_ID in ids:
    ids = ids[:ids.index(text_encoder.EOS_ID)]
  return encoder.decode(ids)

# Melody-conditioned Transformer model
model_name = 'transformer'
hparams_set = 'transformer_tpu'
ckpt_path = '/content/checkpoints/melody_conditioned_model_16.ckpt'

class MelodyToPianoPerformanceProblem(score2perf.AbsoluteMelody2PerfProblem):
  @property
  def add_eos_symbol(self):
    return True

problem = MelodyToPianoPerformanceProblem()
melody_conditioned_encoders = problem.get_feature_encoders()

# Set up HParams.
hparams = trainer_lib.create_hparams(hparams_set=hparams_set)
trainer_lib.add_problem_hparams(hparams, problem)
hparams.num_hidden_layers = 16
hparams.sampling_method = 'random'

# Set up decoding HParams.
decode_hparams = decoding.decode_hparams()
decode_hparams.alpha = 0.0
decode_hparams.beam_size = 1

# Create Estimator.
run_config = trainer_lib.create_run_config(hparams)
estimator = trainer_lib.create_estimator(
    model_name, hparams, run_config,
    decode_hparams=decode_hparams)

# These values will be changed by the following cell.
inputs = []
decode_length = 0

# Create input generator.
def input_generator():
  global inputs
  while True:
    yield {
        'inputs': np.array([[inputs]], dtype=np.int32),
        'targets': np.zeros([1, 0], dtype=np.int32),
        'decode_length': np.array(decode_length, dtype=np.int32)
    }

# Start the Estimator, loading from the specified checkpoint.
input_fn = decoding.make_input_fn_from_generator(input_generator())
melody_conditioned_samples = estimator.predict(
    input_fn, checkpoint_path=ckpt_path)

# "Burn" one.
_ = next(melody_conditioned_samples)

# Extract melody from uploaded MIDI file.
melody_ns = upload_midi()
melody_instrument = mm.infer_melody_for_sequence(melody_ns)
notes = [note for note in melody_ns.notes
    if note.instrument == melody_instrument]
del melody_ns.notes[:]
melody_ns.notes.extend(
sorted(notes, key=lambda note: note.start_time))
for i in range(len(melody_ns.notes) - 1):
    melody_ns.notes[i].end_time = melody_ns.notes[i + 1].start_time
    inputs = melody_conditioned_encoders['inputs'].encode_note_sequence(
    melody_ns)

# Play the melody
mm.play_sequence(
    melody_ns,
    synth=mm.fluidsynth, sample_rate=SAMPLE_RATE, sf2_path=SF2_PATH)
mm.plot_sequence(melody_ns)

# Generate sample events for accompaniment
decode_length = 4096
sample_ids = next(melody_conditioned_samples)['outputs']

# Decode to NoteSequence.
midi_filename = decode(
    sample_ids,
    encoder=melody_conditioned_encoders['targets'])
accompaniment_ns = mm.midi_file_to_note_sequence(midi_filename)

# Play and plot.
mm.play_sequence(
    accompaniment_ns,
    synth=mm.fluidsynth, sample_rate=SAMPLE_RATE, sf2_path=SF2_PATH)
mm.plot_sequence(accompaniment_ns)

# Save generated accompaniment
mm.sequence_proto_to_midi_file(
    accompaniment_ns, '/tmp/accompaniment.mid')
files.download('/tmp/accompaniment.mid')
