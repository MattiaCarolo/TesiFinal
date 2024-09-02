import torch
import numpy as np
import os
import sys
import optparse

import config
from Utils import utils
from Utils.ControlSeq import ControlSeq, Control
from Utils.EventSeq import EventSeq
from Utils.NotesSeq import NoteSeq

from config import device, model as model_config
from model import PerformanceRNN

# pylint: disable=E1101,E1102


# ========================================================================
# Settings
# ========================================================================

def getopt():
    parser = optparse.OptionParser()

    parser.add_option('-c', '--control',
                      dest='control',
                      type='string',
                      default=None,
                      help=('control or a processed data file path, '
                            'e.g., "PITCH_HISTOGRAM;NOTE_DENSITY" like '
                            '"2,0,1,1,0,1,0,1,1,0,0,1;4", or '
                            '";3" (which gives all pitches the same probability), '
                            'or "/path/to/processed/midi/file.data" '
                            '(uses control sequence from the given processed data)'))

    parser.add_option('-b', '--batch-size',
                      dest='batch_size',
                      type='int',
                      default=8)

    parser.add_option('-s', '--session',
                      dest='sess_path',
                      type='string',
                      default='save/train.sess',
                      help='session file containing the trained model')

    parser.add_option('-o', '--output-dir',
                      dest='output_dir',
                      type='string',
                      default='output/')

    parser.add_option('-l', '--max-length',
                      dest='max_len',
                      type='int',
                      default=0)

    parser.add_option('-g', '--greedy-ratio',
                      dest='greedy_ratio',
                      type='float',
                      default=1.0)

    parser.add_option('-B', '--beam-size',
                      dest='beam_size',
                      type='int',
                      default=0)

    parser.add_option('-S', '--stochastic-beam-search',
                      dest='stochastic_beam_search',
                      action='store_true',
                      default=False)

    parser.add_option('-T', '--temperature',
                      dest='temperature',
                      type='float',
                      default=1.0)

    parser.add_option('-z', '--init-zero',
                      dest='init_zero',
                      action='store_true',
                      default=False)

    return parser.parse_args()[0]


opt = getopt()

# ------------------------------------------------------------------------

output_dir = opt.output_dir
sess_path = opt.sess_path
batch_size = opt.batch_size
max_len = opt.max_len
greedy_ratio = opt.greedy_ratio
control = opt.control
use_beam_search = False #opt.beam_size > 0
stochastic_beam_search = opt.stochastic_beam_search
beam_size = opt.beam_size
temperature = opt.temperature
init_zero = opt.init_zero

if use_beam_search:
    greedy_ratio = 'DISABLED'
else:
    beam_size = 'DISABLED'

assert os.path.isfile(sess_path), f'"{sess_path}" is not a file'

if control is not None:
    if os.path.isfile(control) or os.path.isdir(control):
        if os.path.isdir(control):
            files = list(utils.find_files_by_extensions(control))
            assert len(files) > 0, f'no file in "{control}"'
            control = np.random.choice(files)
        _, _,compressed_controls, _ = torch.load(control)
        controls = ControlSeq.recover_compressed_array(compressed_controls)
        if max_len == 0:
            max_len = controls.shape[0]
        controls = torch.tensor(controls, dtype=torch.float32)
        controls = controls.unsqueeze(1).repeat(1, batch_size, 1).to(device)
        control = f'control sequence from "{control}"'

    else:
        pitch_histogram, note_density = control.split(';')
        pitch_histogram = list(filter(len, pitch_histogram.split(',')))
        if len(pitch_histogram) == 0:
            pitch_histogram = np.ones(12) / 12
        else:
            pitch_histogram = np.array(list(map(float, pitch_histogram)))
            assert pitch_histogram.size == 12
            assert np.all(pitch_histogram >= 0)
            pitch_histogram = pitch_histogram / pitch_histogram.sum() \
                if pitch_histogram.sum() else np.ones(12) / 12
        note_density = int(note_density)
        assert note_density in range(len(ControlSeq.note_density_bins))
        control = Control(pitch_histogram, note_density)
        controls = torch.tensor(control.to_array(), dtype=torch.float32)
        controls = controls.repeat(1, batch_size, 1).to(device)
        control = repr(control)

else:
    controls = None
    control = 'NONE'

assert max_len > 0, 'either max length or control sequence length should be given'

# ------------------------------------------------------------------------

print('-' * 70)
print('Session:', sess_path)
print('Batch size:', batch_size)
print('Max length:', max_len)
print('Greedy ratio:', greedy_ratio)
print('Beam size:', beam_size)
print('Beam search stochastic:', stochastic_beam_search)
print('Output directory:', output_dir)
print('Controls:', control)
print('Temperature:', temperature)
print('Init zero:', init_zero)
print('-' * 70)


# ========================================================================
# Generating
# ========================================================================

state = torch.load(sess_path, map_location=device)
model = PerformanceRNN(**state['model_config']).to(device)
model.load_state_dict(state['model_state'])
model.eval()
print(model)
print('-' * 70)

if init_zero:
    init = torch.zeros(batch_size, model.init_dim).to(device)
else:
    init = torch.randn(batch_size, model.init_dim).to(device)

with torch.no_grad():
    if use_beam_search:
        outputs = model.beam_search(init, max_len, beam_size,
                                    controls=controls,
                                    temperatdure=temperature,
                                    stochastic=stochastic_beam_search,
                                    verbose=True)
    else:
        print("bolo")
        outputs = model.generate(init, max_len,
                                 controls=controls,
                                 greedy=greedy_ratio,
                                 temperature=temperature,
                                 verbose=True)

outputs = outputs.cpu().numpy().T  # [batch, steps]
# ========================================================================
# Saving
# ========================================================================
print(outputs)

class Event:

    def __init__(self, type, time, value):
        self.type = type
        self.time = time
        self.value = value

    def __repr__(self):
        return 'Event(type={}, time={}, value={})'.format(
            self.type, self.time, self.value)

def from_array(event_indeces):
    time = 0
    events = []
    for event_index in event_indeces:
        for event_type, feat_range in EventSeq.feat_ranges().items():
            if (feat_range.start <= event_index).any() < feat_range.stop:
                event_value = event_index - feat_range.start
                events.append(Event(event_type, time, event_value))
                if event_type == 'time_shift':
                    time += EventSeq.time_shift_bins[event_value]
                break
    return EventSeq(events)

DEFAULT_SAVING_PROGRAM = 1
DEFAULT_LOADING_PROGRAMS = range(128)
DEFAULT_RESOLUTION = 220
DEFAULT_TEMPO = 120
DEFAULT_VELOCITY = 64
DEFAULT_PITCH_RANGE = range(21, 109)
DEFAULT_VELOCITY_RANGE = range(21, 109)
DEFAULT_NORMALIZATION_BASELINE = 60  # C4

USE_VELOCITY = True
BEAT_LENGTH = 60 / DEFAULT_TEMPO
DEFAULT_TIME_SHIFT_BINS = 1.15 ** np.arange(32) / 65
DEFAULT_VELOCITY_STEPS = 32
DEFAULT_NOTE_LENGTH = BEAT_LENGTH * 2
MIN_NOTE_LENGTH = BEAT_LENGTH / 2

from Utils.NotesSeq import Note, NoteSeq

def to_note_seq(events):
    time = 0
    notes = []

    velocity = DEFAULT_VELOCITY
    velocity_bins = EventSeq.get_velocity_bins()

    last_notes = {}

    for event in events:
        if event.type == 'note_on':
            pitch = event.value + EventSeq.pitch_range.start
            note = Note(velocity, pitch, time, None)
            notes.append(note)
            last_notes[pitch] = note

        elif event.type == 'note_off':
            pitch = event.value + EventSeq.pitch_range.start

            if pitch in last_notes:
                note = last_notes[pitch]
                note.end = max(time, note.start + MIN_NOTE_LENGTH)
                del last_notes[pitch]

        elif event.type == 'velocity':
            index = min(event.value, velocity_bins.size - 1)
            velocity = velocity_bins[index]

        elif event.type == 'time_shift':
            time += EventSeq.time_shift_bins[event.value]

    for note in notes:
        if note.end is None:
            note.end = note.start + DEFAULT_NOTE_LENGTH

        note.velocity = int(note.velocity)

    return NoteSeq(notes)

def event_indeces_to_midi_file(event_indeces, midi_file_name, velocity_scale=0.8):
    event_seq = from_array(event_indeces)
    print(event_seq)
    note_seq = event_seq.to_note_seq()
    for note in note_seq.notes:
        note.velocity = int((note.velocity - 64) * velocity_scale + 64)
    print(note_seq)
    note_seq.to_midi_file(midi_file_name)
    return len(note_seq.notes)

os.makedirs(output_dir, exist_ok=True)

for i, output in enumerate(outputs):
    name = f'output-{i:03d}.mid'
    path = os.path.join(output_dir, name)
    n_notes = utils.event_indeces_to_midi_file(output, path)
    print(f'===> {path} ({n_notes} notes)')
