import numpy as np
import copy
import itertools
import collections
from Utils.NotesSeq import *
from pretty_midi import PrettyMIDI, Note, Instrument

# ==================================================================================
# Parameters
# ==================================================================================

# EventSeq ------------------------------------------------------------------------

USE_VELOCITY = True
BEAT_LENGTH = 60 / DEFAULT_TEMPO
DEFAULT_TIME_SHIFT_BINS = 1.15 ** np.arange(32) / 65
DEFAULT_VELOCITY_STEPS = 32
DEFAULT_NOTE_LENGTH = BEAT_LENGTH * 2
MIN_NOTE_LENGTH = BEAT_LENGTH / 2

# ==================================================================================
# Events
# ==================================================================================

class Event:
    # •	Purpose: Represents an individual event in a sequence. An event can be a note being played, a change in velocity, or a time shift.
	# •	Attributes:
	    # •	type: The type of the event (e.g., ‘note_on’, ‘note_off’, ‘velocity’, ‘time_shift’).
	    # •	time: The time at which the event occurs.
	    # •	value: The value associated with the event (e.g., pitch for a ‘note_on’ event, velocity level for a ‘velocity’ event, etc.).
	    # •	__repr__ Method: Provides a string representation of the event for easier debugging and logging.

    def __init__(self, type, time, value):
        self.type = type
        self.time = time
        self.value = value

    def __repr__(self):
        return 'Event(type={}, time={}, value={})'.format(
            self.type, self.time, self.value)


class EventSeq:

    pitch_range = DEFAULT_PITCH_RANGE
    velocity_range = DEFAULT_VELOCITY_RANGE
    velocity_steps = DEFAULT_VELOCITY_STEPS
    time_shift_bins = DEFAULT_TIME_SHIFT_BINS

    @staticmethod
    def from_note_seq(note_seq):
    # •	Purpose: Converts a NoteSeq object into an EventSeq object.
	# •	Parameters:
	#   •	note_seq: A NoteSeq object containing a list of Note objects.
	# •	Functionality:
	#   •	Initializes an empty list note_events to store events derived from the notes.
	#   •	If velocity is enabled (USE_VELOCITY), calculates velocity_bins for quantizing velocities into discrete steps.
	#   •	Iterates over the notes in note_seq:
	#   •	Creates ‘velocity’ events if USE_VELOCITY is enabled.
	#   •	Creates ‘note_on’ and ‘note_off’ events for each note, using its start and end times.
	#   •	Sorts the events by time.
	#   •	Adds ‘time_shift’ events to represent the time intervals between consecutive events.
	#   •	Returns an EventSeq object containing all the events.
        note_events = []

        if USE_VELOCITY:
            velocity_bins = EventSeq.get_velocity_bins()

        for note in note_seq.notes:
            if note.pitch in EventSeq.pitch_range:
                if USE_VELOCITY:
                    velocity = note.velocity
                    velocity = max(velocity, EventSeq.velocity_range.start)
                    velocity = min(velocity, EventSeq.velocity_range.stop - 1)
                    velocity_index = np.searchsorted(velocity_bins, velocity)
                    note_events.append(Event('velocity', note.start, velocity_index))

                pitch_index = note.pitch - EventSeq.pitch_range.start
                note_events.append(Event('note_on', note.start, pitch_index))
                note_events.append(Event('note_off', note.end, pitch_index))

        note_events.sort(key=lambda event: event.time)  # stable

        #print(note_events)

        events = []

        for i, event in enumerate(note_events):
            events.append(event)
            #print(event)

            if event is note_events[-1]:
                break

            interval = note_events[i + 1].time - event.time
            shift = 0

            while interval - shift >= EventSeq.time_shift_bins[0]:
                index = np.searchsorted(EventSeq.time_shift_bins,
                                        interval - shift, side='right') - 1
                events.append(Event('time_shift', event.time + shift, index))
                shift += EventSeq.time_shift_bins[index]

        return EventSeq(events)

    @staticmethod
    def from_array(event_indeces):
    # •	Purpose: Creates an EventSeq object from an array of event indices.
	# •	Parameters:
	#   •	event_indeces: An array of integers representing event indices.
	# •	Functionality:
	#   •	Iterates over the event indices, determining the type and value of each event.
	#   •	Converts each index into an Event object, updating the time if the event is a ‘time_shift’.
	#   •	Returns an EventSeq object containing all the events.

        time = 0
        events = []
        for event_index in event_indeces:
            for event_type, feat_range in EventSeq.feat_ranges().items():
                if feat_range.start <= event_index < feat_range.stop:
                    event_value = event_index - feat_range.start
                    events.append(Event(event_type, time, event_value))
                    if event_type == 'time_shift':
                        time += EventSeq.time_shift_bins[event_value]
                    break

        return EventSeq(events)

    @staticmethod
    def dim():
    # •	Purpose: Returns the total dimension of all possible event types.
	# •	Functionality:
	#   •	Sums up the dimensions of all features (note_on, note_off, velocity, time_shift).
        return sum(EventSeq.feat_dims().values())

    @staticmethod
    def feat_dims():
    # •	Purpose: Returns the dimensions of each feature type.
	# •	Functionality:
	#   •	Calculates the number of discrete values each feature (event type) can take.
	#   •	Returns an ordered dictionary with feature types and their respective dimensions.
        feat_dims = collections.OrderedDict()
        feat_dims['note_on'] = len(EventSeq.pitch_range)
        feat_dims['note_off'] = len(EventSeq.pitch_range)
        if USE_VELOCITY:
            feat_dims['velocity'] = EventSeq.velocity_steps
        feat_dims['time_shift'] = len(EventSeq.time_shift_bins)
        return feat_dims

    @staticmethod
    def feat_ranges():
    # •	Purpose: Returns the ranges of indices corresponding to each feature type.
	# •	Functionality:
	#   •	Computes the range of indices for each event type based on their dimensions.
	#   •	Returns an ordered dictionary mapping feature names to their ranges.
        offset = 0
        feat_ranges = collections.OrderedDict()
        for feat_name, feat_dim in EventSeq.feat_dims().items():
            feat_ranges[feat_name] = range(offset, offset + feat_dim)
            offset += feat_dim
        return feat_ranges

    @staticmethod
    def get_velocity_bins():
    # •	Purpose: Computes the bins for quantizing velocities.
	# •	Functionality:
	#   •	Generates an array of equally spaced values between the minimum and maximum velocity, corresponding to the number of velocity steps.
        n = EventSeq.velocity_range.stop - EventSeq.velocity_range.start
        return np.arange(EventSeq.velocity_range.start,
                         EventSeq.velocity_range.stop,
                         n / (EventSeq.velocity_steps - 1))

    def __init__(self, events=[]):
    # •	Purpose: Initializes an EventSeq object.
	# •	Parameters:
	#   •	events: A list of Event objects.
	# •	Functionality:
	#   •	Creates a deep copy of the events to avoid modifying the original list.
	#   •	Recomputes the event times, taking into account ‘time_shift’ events.
        #for event in events:
            #print("oggetto di tipo = " + str(type(event)))
        #    assert isinstance(event, Event)

        self.events = copy.deepcopy(events)

        # compute event times again to consider time_shift
        time = 0
        for event in self.events:
            event.time = time
            if event.type == 'time_shift':
                time += EventSeq.time_shift_bins[event.value]

    def to_note_seq(self):
    # •	Purpose: Converts the EventSeq back to a NoteSeq.
	# •	Functionality:
	#   •	Iterates over the events and reconstructs the corresponding notes.
	#   •	Manages velocity changes, time shifts, and note durations.
	#   •	Returns a new NoteSeq object containing the reconstructed notes.
        time = 0
        notes = []

        velocity = DEFAULT_VELOCITY
        velocity_bins = EventSeq.get_velocity_bins()

        last_notes = {}

        for event in self.events:
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

    def to_array(self):
    # •	Purpose: Converts the EventSeq object into an array of integer indices representing the events.
	# •	Functionality:
	#   •	Uses the feature ranges (feat_ranges) to map each event type and value to a unique integer index.
	#   •	Creates a list of these indices for all events in the sequence (self.events).
	#   •	Determines the appropriate data type (np.uint8 or np.uint16) for the array based on the total number of possible events (dim), ensuring memory efficiency.
	#   •	Returns a NumPy array containing these integer indices.
        feat_idxs = EventSeq.feat_ranges()
        idxs = [feat_idxs[event.type][event.value] for event in self.events]
        dtype = np.uint8 if EventSeq.dim() <= 256 else np.uint16
        return np.array(idxs, dtype=dtype)
