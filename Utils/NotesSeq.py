import numpy as np
import copy
import itertools
import collections
from pretty_midi import PrettyMIDI, Note, Instrument

# ==================================================================================
# Parameters
# ==================================================================================

# Source comes from process.py from PerformanceRNN

# NoteSeq -------------------------------------------------------------------------

DEFAULT_SAVING_PROGRAM = 1
DEFAULT_LOADING_PROGRAMS = range(128)
DEFAULT_RESOLUTION = 220
DEFAULT_TEMPO = 120
DEFAULT_VELOCITY = 64
DEFAULT_PITCH_RANGE = range(21, 109)
DEFAULT_VELOCITY_RANGE = range(21, 109)
DEFAULT_NORMALIZATION_BASELINE = 60  # C4

class NoteSeq:

    @staticmethod
    def from_midi(midi, programs=DEFAULT_LOADING_PROGRAMS):

        notes = itertools.chain(*[
            inst.notes for inst in midi.instruments
            if inst.program in programs and not inst.is_drum])
        #print(notes)
        return NoteSeq(list(notes))

    @staticmethod
    def from_tracks(midi, flag, programs=DEFAULT_LOADING_PROGRAMS):
        # Variable flag permits to manage different tracks
        group_indices = flag
        group_notes = itertools.chain(*[
            midi.instruments[i].notes for i in group_indices
        ])
        #print(list(group_notes))
        return NoteSeq(list(group_notes))


    @staticmethod
    def from_midi_file(path, *args, **kwargs):
        midi = PrettyMIDI(path)
        track_len = len(midi.instruments)

        if(track_len < 3):
            stream1 = NoteSeq.from_tracks(midi,[0])
            stream2 = NoteSeq.from_tracks(midi,[1])
            #print(" 2 tracce : " + path)
        elif(track_len < 4):
            stream1 = NoteSeq.from_tracks(midi,[0])
            stream2 = NoteSeq.from_tracks(midi,[1,2])
            #print(" 3 tracce : " + path)
        else:
            stream1 = NoteSeq.from_tracks(midi,[0,1])
            stream2 = NoteSeq.from_tracks(midi,[2,3])
            #print(" 4 tracce : " + path)


        return stream1,stream2
        #return NoteSeq.from_midi(midi, *args, **kwargs)

    @staticmethod
    def merge(*note_seqs):
        notes = itertools.chain(*[seq.notes for seq in note_seqs])
        return NoteSeq(list(notes))

    def __init__(self, notes=[]):
        self.notes = []
        if notes:
            for note in notes:
                assert isinstance(note, Note)
            notes = filter(lambda note: note.end >= note.start, notes)
            self.add_notes(list(notes))

    def copy(self):
        return copy.deepcopy(self)

    def to_midi(self, program=DEFAULT_SAVING_PROGRAM,
                resolution=DEFAULT_RESOLUTION, tempo=DEFAULT_TEMPO):
        midi = PrettyMIDI(resolution=resolution, initial_tempo=tempo)
        inst = Instrument(program, False, 'NoteSeq')
        inst.notes = copy.deepcopy(self.notes)
        midi.instruments.append(inst)
        return midi

    def to_midi_file(self, path, *args, **kwargs):
        self.to_midi(*args, **kwargs).write(path)

    def add_notes(self, notes):
        self.notes += notes
        self.notes.sort(key=lambda note: note.start)

    def adjust_pitches(self, offset):
        for note in self.notes:
            pitch = note.pitch + offset
            pitch = 0 if pitch < 0 else pitch
            pitch = 127 if pitch > 127 else pitch
            note.pitch = pitch

    def adjust_velocities(self, offset):
        for note in self.notes:
            velocity = note.velocity + offset
            velocity = 0 if velocity < 0 else velocity
            velocity = 127 if velocity > 127 else velocity
            note.velocity = velocity

    def adjust_time(self, offset):
        for note in self.notes:
            note.start += offset
            note.end += offset

    def trim_overlapped_notes(self, min_interval=0):
        last_notes = {}
        for i, note in enumerate(self.notes):
            if note.pitch in last_notes:
                last_note = last_notes[note.pitch]
                if note.start - last_note.start <= min_interval:
                    last_note.end = max(note.end, last_note.end)
                    last_note.velocity = max(note.velocity, last_note.velocity)
                    del self.notes[i]
                elif note.start < last_note.end:
                    last_note.end = note.start
            else:
                last_notes[note.pitch] = note

