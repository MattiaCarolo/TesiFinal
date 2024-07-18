import numpy as np
import copy
import itertools
import collections
from Utils.NotesSeq import *
from Utils.EventSeq import *
from pretty_midi import PrettyMIDI, Note, Instrument

# ==================================================================================
# Parameters
# ==================================================================================

# ControlSeq ----------------------------------------------------------------------

DEFAULT_WINDOW_SIZE = BEAT_LENGTH * 4
DEFAULT_NOTE_DENSITY_BINS = np.arange(12) * 3 + 1


# ==================================================================================
# Controls
# ==================================================================================

class Control:

    def __init__(self, pitch_histogram, note_density):
        self.pitch_histogram = pitch_histogram  # list
        self.note_density = note_density  # int

    def __repr__(self):
        return 'Control(pitch_histogram={}, note_density={})'.format(
            self.pitch_histogram, self.note_density)

    def to_array(self):
        feat_dims = ControlSeq.feat_dims()
        ndens = np.zeros([feat_dims['note_density']])
        ndens[self.note_density] = 1.  # [dens_dim]
        phist = np.array(self.pitch_histogram)  # [hist_dim]
        return np.concatenate([ndens, phist], 0)  # [dens_dim + hist_dim]


class ControlSeq:

    note_density_bins = DEFAULT_NOTE_DENSITY_BINS
    window_size = DEFAULT_WINDOW_SIZE

    @staticmethod
    def from_event_seq(event_seq):
        events = list(event_seq.events)
        start, end = 0, 0

        pitch_count = np.zeros([12])
        note_count = 0

        controls = []

        def _rel_pitch(pitch):
            return (pitch - 24) % 12

        for i, event in enumerate(events):

            while start < i:
                if events[start].type == 'note_on':
                    abs_pitch = events[start].value + EventSeq.pitch_range.start
                    rel_pitch = _rel_pitch(abs_pitch)
                    pitch_count[rel_pitch] -= 1.
                    note_count -= 1.
                start += 1

            while end < len(events):
                if events[end].time - event.time > ControlSeq.window_size:
                    break
                if events[end].type == 'note_on':
                    abs_pitch = events[end].value + EventSeq.pitch_range.start
                    rel_pitch = _rel_pitch(abs_pitch)
                    pitch_count[rel_pitch] += 1.
                    note_count += 1.
                end += 1

            pitch_histogram = (
                pitch_count / note_count
                if note_count
                else np.ones([12]) / 12
            ).tolist()

            note_density = max(np.searchsorted(
                ControlSeq.note_density_bins,
                note_count, side='right') - 1, 0)

            controls.append(Control(pitch_histogram, note_density))

        return ControlSeq(controls)

    @staticmethod
    def dim():
        return sum(ControlSeq.feat_dims().values())

    @staticmethod
    def feat_dims():
        note_density_dim = len(ControlSeq.note_density_bins)
        return collections.OrderedDict([
            ('pitch_histogram', 12),
            ('note_density', note_density_dim)
        ])

    @staticmethod
    def feat_ranges():
        offset = 0
        feat_ranges = collections.OrderedDict()
        for feat_name, feat_dim in ControlSeq.feat_dims().items():
            feat_ranges[feat_name] = range(offset, offset + feat_dim)
            offset += feat_dim
        return feat_ranges

    @staticmethod
    def recover_compressed_array(array):
        feat_dims = ControlSeq.feat_dims()
        assert array.shape[1] == 1 + feat_dims['pitch_histogram']
        ndens = np.zeros([array.shape[0], feat_dims['note_density']])
        ndens[np.arange(array.shape[0]), array[:, 0]] = 1.  # [steps, dens_dim]
        phist = array[:, 1:].astype(np.float64) / 255  # [steps, hist_dim]
        return np.concatenate([ndens, phist], 1)  # [steps, dens_dim + hist_dim]

    def __init__(self, controls):
        for control in controls:
            assert isinstance(control, Control)
        self.controls = copy.deepcopy(controls)

    def to_compressed_array(self):
        ndens = [control.note_density for control in self.controls]
        ndens = np.array(ndens, dtype=np.uint8).reshape(-1, 1)
        phist = [control.pitch_histogram for control in self.controls]
        phist = (np.array(phist) * 255).astype(np.uint8)
        return np.concatenate([
            ndens,  # [steps, 1] density index
            phist  # [steps, hist_dim] 0-255
        ], 1)  # [steps, hist_dim + 1]
