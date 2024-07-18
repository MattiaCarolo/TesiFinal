import os
import re
import sys
import torch
import hashlib
import itertools
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


from progress.bar import Bar
from concurrent.futures import ProcessPoolExecutor

from Utils.NotesSeq import NoteSeq as ns
from Utils.EventSeq import EventSeq as es
from Utils.ControlSeq import ControlSeq as cs
from Utils import utils

import warnings
warnings.filterwarnings("ignore")

def preprocess_midi(path):

    note_seq = ns.from_midi_file(path)
    iter = itertools.cycle(note_seq)

    es1 = list
    es2 = list

    cs1 = list
    cs2 = list
    ev = True
    #print("evaluating")
    for seq in iter:
        #print(seq)
        if ev:
            #print("1st iter")
            seq.adjust_time(-seq.notes[0].start)
            event_seq = es.from_note_seq(seq)
            control_seq = cs.from_event_seq(event_seq)
            es1 = event_seq.to_array()
            cs1 = control_seq.to_compressed_array()
            ev = False
        else:
            #print("2nd iter")
            seq.adjust_time(-seq.notes[0].start)
            event_seq = es.from_note_seq(seq)
            control_seq = cs.from_event_seq(event_seq)
            es2 = event_seq.to_array()
            cs2 = control_seq.to_compressed_array()
            break

    return es1, es2, cs1, cs2

def preprocess_midi_files_under(midi_root, save_dir, num_workers):
    midi_paths = list(utils.find_files_by_extensions(midi_root, ['.mid', '.midi']))
    os.makedirs(save_dir, exist_ok=True)
    out_fmt = '{}-{}.data'
    
    results = []
    num_cores = os.cpu_count()
    print(f"Number of CPU cores: {num_cores}")
    print(f"Setting number of workers to : {num_cores}")

    executor = ProcessPoolExecutor(num_cores)

    for path in midi_paths:
        try:
            results.append((path, executor.submit(preprocess_midi, path)))
        except KeyboardInterrupt:
            print(' Abort')
            return
        except:
            print(' Error')
            continue

    for path, future in Bar('Processing').iter(results):
        print(' ', end='[{}]'.format(path), flush=True)
        name = os.path.basename(path)
        code = hashlib.md5(path.encode()).hexdigest()
        save_path = os.path.join(save_dir, out_fmt.format(name, code))
        torch.save(future.result(), save_path)

    print('Done')


def preprocess_midi_files_under_noconc(midi_root, save_dir, num_workers):
    midi_paths = list(utils.find_files_by_extensions(midi_root, ['.mid', '.midi']))
    os.makedirs(save_dir, exist_ok=True)
    out_fmt = '{}-{}.data'
    
    results = []

    for path in Bar('Processing').iter(midi_paths):
        print("Running")
        try:
            results.append((path, preprocess_midi(path)))
        except KeyboardInterrupt:
            print(' Abort')
            return
        except:
            print(' Error')
            continue

    for path, future in Bar('Saving').iter(results):
        print(' ', end='[{}]'.format(path), flush=True)
        name = os.path.basename(path)
        code = hashlib.md5(path.encode()).hexdigest()
        save_path = os.path.join(save_dir, out_fmt.format(name, code))
        torch.save(future, save_path)

    print('Done')


if __name__ == '__main__':

    preprocess_midi_files_under(
            midi_root=sys.argv[1],
            save_dir=sys.argv[2],
            num_workers=int(sys.argv[3]))
