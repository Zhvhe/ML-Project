print('Copying checkpoints and Salamander piano SoundFont (via https://sites.google.com/site/soundfonts4u) from GCS...')
!gsutil -q -m cp -r gs://magentadata/models/music_transformer/* /content/
!gsutil -q -m cp gs://magentadata/soundfonts/Yamaha-C5-Salamander-JNv5.1.sf2 /content/

print('Installing dependencies...')
!apt-get update -qq && apt-get install -qq libfluidsynth1 build-essential libasound2-dev libjack-dev
!pip install -qU magenta pyfluidsynth

import ctypes.util
def proxy_find_library(lib):
  if lib == 'fluidsynth':
    return 'libfluidsynth.so.1'
  else:
    return ctypes.util.find_library(lib)
ctypes.util.find_library = proxy_find_library

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

model_name = 'transformer'
hparams_set = 'transformer_tpu'
ckpt_path = '/content/checkpoints/unconditional_model_16.ckpt'

class PianoPerformanceLanguageModelProblem(score2perf.Score2PerfProblem):
  @property
  def add_eos_symbol(self):
    return True

problem = PianoPerformanceLanguageModelProblem()
unconditional_encoders = problem.get_feature_encoders()

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

# Create input generator (so we can adjust priming and
# decode length on the fly).
def input_generator():
  global targets
  global decode_length
  while True:
    yield {
        'targets': np.array([targets], dtype=np.int32),
        'decode_length': np.array(decode_length, dtype=np.int32)
    }

# These values will be changed by subsequent cells.
targets = []
decode_length = 0

# Start the Estimator, loading from the specified checkpoint.
input_fn = decoding.make_input_fn_from_generator(input_generator())
unconditional_samples = estimator.predict(
    input_fn, checkpoint_path=ckpt_path)

# "Burn" one.
_ = next(unconditional_samples)

targets = []
decode_length = 1024

# Generate sample events.
sample_ids = next(unconditional_samples)['outputs']

# Decode to NoteSequence.
midi_filename = decode(
    sample_ids,
    encoder=unconditional_encoders['targets'])
unconditional_ns = mm.midi_file_to_note_sequence(midi_filename)

# Play and plot.
mm.play_sequence(
    unconditional_ns,
    synth=mm.fluidsynth, sample_rate=SAMPLE_RATE, sf2_path=SF2_PATH)
mm.plot_sequence(unconditional_ns)

# Download MIDI and play
mm.sequence_proto_to_midi_file(
    unconditional_ns, '/tmp/unconditional.mid')
files.download('/tmp/unconditional.mid')
