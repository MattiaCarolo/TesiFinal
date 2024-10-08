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

# Parameters for handling MIDI notes
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
    #     Purpose: Creates a NoteSeq object from a PrettyMIDI object (representing a MIDI file).
	# •	Parameters:
	#   •	midi: A PrettyMIDI object that represents a MIDI file.
	#   •	programs: A list of instrument programs to include (default is all programs from 0 to 127).
	# •	Functionality:
	#   •	Extracts notes from all instruments in the midi object that match the specified programs and are not drums.
	#   •	Combines these notes into a single sequence using itertools.chain.
	#   •	Returns a NoteSeq object containing all these notes.
        notes = itertools.chain(*[
            inst.notes for inst in midi.instruments
            if inst.program in programs and not inst.is_drum])
        #print(notes)
        return NoteSeq(list(notes))

    @staticmethod
    def from_tracks(midi, flag, programs=DEFAULT_LOADING_PROGRAMS):
    #     Purpose: Creates a NoteSeq object from specific tracks (instruments) of a MIDI file.
	# •	Parameters:
	#   •	midi: A PrettyMIDI object.
	#   •	flag: A list of indices indicating which tracks (instruments) to include.
    #           Cases can be found on function 'from_midi_file'
	#   •	programs: List of instrument programs (defaults to all programs).
	# •	Functionality:
	#   •	Extracts notes from the specified tracks in the flag list.
	#   •	Combines these notes and returns a NoteSeq object containing them.
    # Variable flag permits to manage different tracks
        group_indices = flag
        group_notes = itertools.chain(*[
            midi.instruments[i].notes for i in group_indices
        ])
        #print(list(group_notes))
        return NoteSeq(list(group_notes))


    @staticmethod
    def from_midi_file(path, *args, **kwargs):
    # •	Purpose: Creates two NoteSeq objects from a MIDI file specified by a file path.
	# •	Parameters:
	#   •	path: The file path to the MIDI file.
	#   •	Additional args and kwargs to pass to the PrettyMIDI object.
	# •	Functionality:
	#   •	Reads the MIDI file from the given path.
	#   •	Depending on the number of tracks, it selects specific tracks to create two NoteSeq objects.
	# •	Returns these two NoteSeq objects under two different streams

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
    # •	Purpose: Merges multiple NoteSeq objects into a single NoteSeq object (TLDR )
	# •	Parameters:
	#   •	note_seqs: A variable number of NoteSeq objects to be merged.
	# •	Functionality:
	#   •	Combines all the notes from the given NoteSeq objects.
	# •	Returns a new NoteSeq object containing all these notes concatenated
        notes = itertools.chain(*[seq.notes for seq in note_seqs])
        return NoteSeq(list(notes))

    def __init__(self, notes=[]):
    # •	Purpose: Initializes a NoteSeq object.
	# •	Parameters:
	#   •	notes: A list of Note objects (from pretty_midi).
	# •	Functionality:
	#   •	Initializes the notes attribute to an empty list.
	# •	If notes are provided, it checks that each note is a valid Note object and that the note’s end time is after its start time.
	# •	Adds valid notes to the sequence.
        self.notes = []
        if notes:
            for note in notes:
                assert isinstance(note, Note)
            notes = filter(lambda note: note.end >= note.start, notes)
            self.add_notes(list(notes))

    def copy(self):
    # •	Purpose: Creates a deep copy of the NoteSeq object.
	# •	Functionality:
	# •	Returns a new NoteSeq object that is a deep copy of the current one.
        return copy.deepcopy(self)

    def to_midi(self, program=DEFAULT_SAVING_PROGRAM,
                resolution=DEFAULT_RESOLUTION, tempo=DEFAULT_TEMPO):
    # •	Purpose: Converts the NoteSeq object into a PrettyMIDI object.
	# •	Parameters:
	#   •	program: Instrument program number (default 1).
	#   •	resolution: Resolution of the MIDI file (default 220).
	#   •	tempo: Tempo of the MIDI file (default 120 BPM).
	# •	Functionality:
	#   •	Creates a new PrettyMIDI object with the specified resolution and tempo.
	#   •	Adds the notes from the NoteSeq to an instrument with the given program number.
	# •	Returns the PrettyMIDI object.
        midi = PrettyMIDI(resolution=resolution, initial_tempo=tempo)
        inst = Instrument(program, False, 'NoteSeq')
        inst.notes = copy.deepcopy(self.notes)
        midi.instruments.append(inst)
        return midi

    def to_midi_file(self, path, *args, **kwargs):
    # •	Purpose: Writes the NoteSeq object to a MIDI file.
	# •	Parameters:
	#   •	path: File path where the MIDI file will be saved.
	# •	Functionality:
	#   •	Converts the NoteSeq to a PrettyMIDI object and writes it to the specified file path.
        self.to_midi(*args, **kwargs).write(path)

    def add_notes(self, notes):
    # •	Purpose: Adds notes to the NoteSeq object.
	# •	Parameters:
	#   •	notes: A list of Note objects to add.
	# •	Functionality:
	#   •	Adds the provided notes to the existing list of notes.
	#   •	Sorts the notes by their start time.
        self.notes += notes
        self.notes.sort(key=lambda note: note.start)

    def adjust_pitches(self, offset):
    # •	Purpose: Adjusts the pitch of all notes in the NoteSeq.
	# •	Parameters:
	#   •	offset: An integer offset to add to each note’s pitch.
	# •	Functionality:
	#   •	Increases or decreases each note’s pitch by the offset.
	#   •	Ensures the pitch stays within the valid MIDI range (0-127).
        for note in self.notes:
            pitch = note.pitch + offset
            pitch = 0 if pitch < 0 else pitch
            pitch = 127 if pitch > 127 else pitch
            note.pitch = pitch

    def adjust_velocities(self, offset):
    # •	Purpose: Adjusts the velocity (volume) of all notes in the NoteSeq.
	# •	Parameters:
	#   •	offset: An integer offset to add to each note’s velocity.
	# •	Functionality:
	#   •	Increases or decreases each note’s velocity by the offset.
	#   •	Ensures the velocity stays within the valid MIDI range (0-127).

        for note in self.notes:
            velocity = note.velocity + offset
            velocity = 0 if velocity < 0 else velocity
            velocity = 127 if velocity > 127 else velocity
            note.velocity = velocity

    def adjust_time(self, offset):
    # •	Purpose: Adjusts the start and end times of all notes in the NoteSeq.
	# •	Parameters:
	#   •	offset: A time offset (in seconds) to add to each note’s start and end time.
	# •	Functionality:
	#   •	Increases each note’s start and end times by the offset.
        for note in self.notes:
            note.start += offset
            note.end += offset

    def trim_overlapped_notes(self, min_interval=0):
    # - Purpose: Removes or trims notes that overlap in time.
    # - Parameters:
    #     - `min_interval`: Minimum allowed time interval between overlapping notes (default is 0).
    # - Functionality:
    #   - Iterates through all notes and checks for overlaps with notes of the same pitch.
    #   - If two notes overlap or are closer than `min_interval`, it adjusts or removes notes to resolve the overlap.
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

