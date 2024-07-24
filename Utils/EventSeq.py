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
        return sum(EventSeq.feat_dims().values())

    @staticmethod
    def feat_dims():
        feat_dims = collections.OrderedDict()
        feat_dims['note_on'] = len(EventSeq.pitch_range)
        feat_dims['note_off'] = len(EventSeq.pitch_range)
        if USE_VELOCITY:
            feat_dims['velocity'] = EventSeq.velocity_steps
        feat_dims['time_shift'] = len(EventSeq.time_shift_bins)
        return feat_dims

    @staticmethod
    def feat_ranges():
        offset = 0
        feat_ranges = collections.OrderedDict()
        for feat_name, feat_dim in EventSeq.feat_dims().items():
            feat_ranges[feat_name] = range(offset, offset + feat_dim)
            offset += feat_dim
        return feat_ranges

    @staticmethod
    def get_velocity_bins():
        n = EventSeq.velocity_range.stop - EventSeq.velocity_range.start
        return np.arange(EventSeq.velocity_range.start,
                         EventSeq.velocity_range.stop,
                         n / (EventSeq.velocity_steps - 1))

    def __init__(self, events=[]):
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
        feat_idxs = EventSeq.feat_ranges()
        idxs = [feat_idxs[event.type][event.value] for event in self.events]
        dtype = np.uint8 if EventSeq.dim() <= 256 else np.uint16
        return np.array(idxs, dtype=dtype)
