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
    # •	Purpose: Converts an EventSeq into a ControlSeq by calculating control features such as pitch histograms and note densities over moving windows.
	# •	Parameters:
	#   •	event_seq: An instance of EventSeq, which contains a list of musical events derived from a NoteSeq.
	# •	Functionality:
	#   •	Iterates over the events in the EventSeq to calculate musical features within a moving window (window_size).
	#   •	Computes:
	#       •	Pitch Histogram: A normalized distribution of the 12 pitch classes (C, C#, D, …, B) based on the notes in the window.
	#       •	Note Density: The number of notes within the window, quantized to a predefined set of bins (note_density_bins).
	#   •	Creates and appends Control objects for each window, representing the musical features.
	# •	Returns a ControlSeq object containing the list of computed Control objects.
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
    # •	Purpose: Calculates the total number of dimensions for the control features.
	# •	Functionality:
	# •	Sums up the dimensions of the individual control features (pitch_histogram and note_density).
        return sum(ControlSeq.feat_dims().values())

    @staticmethod
    def feat_dims():
    # •	Purpose: Provides the dimensions for each feature type in the ControlSeq.
	# •	Functionality:
	#   •	Defines the number of bins for the pitch_histogram (12 pitch classes) and note_density (number of bins in note_density_bins).
	#   •	Returns an ordered dictionary containing the feature names and their corresponding dimensions.
        note_density_dim = len(ControlSeq.note_density_bins)
        return collections.OrderedDict([
            ('pitch_histogram', 12),
            ('note_density', note_density_dim)
        ])

    @staticmethod
    def feat_ranges():
    # •	Purpose: Provides the index ranges for each feature type.
	# •	Functionality:
	# •	Computes the range of indices for each control feature type (pitch histogram and note density).
	# •	Returns an ordered dictionary mapping feature names to their index ranges.
        offset = 0
        feat_ranges = collections.OrderedDict()
        for feat_name, feat_dim in ControlSeq.feat_dims().items():
            feat_ranges[feat_name] = range(offset, offset + feat_dim)
            offset += feat_dim
        return feat_ranges

    @staticmethod
    def recover_compressed_array(array):
    # •	Purpose: Recovers the original control features from a compressed array format.
	# •	Parameters:
	#   •	array: A compressed NumPy array where the first column represents note density and the remaining columns represent the pitch histogram.
	# •	Functionality:
	#   •	Validates the array dimensions against the expected feature sizes.
	#   •	Recovers the note density as a one-hot encoded vector.
	#   •	Normalizes the pitch histogram values from the compressed format (0-255) back to their original range (0-1).
	# •	Returns the concatenated array of note density and pitch histogram.
        feat_dims = ControlSeq.feat_dims()
        assert array.shape[1] == 1 + feat_dims['pitch_histogram']
        ndens = np.zeros([array.shape[0], feat_dims['note_density']])
        ndens[np.arange(array.shape[0]), array[:, 0]] = 1.  # [steps, dens_dim]
        phist = array[:, 1:].astype(np.float64) / 255  # [steps, hist_dim]
        return np.concatenate([ndens, phist], 1)  # [steps, dens_dim + hist_dim]

    def __init__(self, controls):
    # •	Purpose: Initializes a ControlSeq object with a list of Control objects.
	# •	Parameters:
	#   •	controls: A list of Control objects representing the musical controls for different windows.
	# •	Functionality:
	#   •	Validates that each element in the controls list is an instance of the Control class.
	#   •	Creates a deep copy of the controls to ensure the original data is not modified.
        for control in controls:
            assert isinstance(control, Control)
        self.controls = copy.deepcopy(controls)

    def to_compressed_array(self):
    # •	Purpose: Converts the ControlSeq into a compressed array format suitable for storage or use in machine learning models.
	# •	Functionality:
	#   •	Extracts the note density and pitch histogram from each Control in the sequence.
	#   •	Converts the note density to a NumPy array and reshapes it to a column vector.
	#   •	Scales the pitch histogram values to the range 0-255 for compression.
	#   •	Concatenates the note density and pitch histogram arrays to create a compressed representation of the control features.
        ndens = [control.note_density for control in self.controls]
        ndens = np.array(ndens, dtype=np.uint8).reshape(-1, 1)
        phist = [control.pitch_histogram for control in self.controls]
        phist = (np.array(phist) * 255).astype(np.uint8)
        return np.concatenate([
            ndens,  # [steps, 1] density index
            phist  # [steps, hist_dim] 0-255
        ], 1)  # [steps, hist_dim + 1]
