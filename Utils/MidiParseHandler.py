import contextlib
import os
import concurrent.futures
import multiprocessing
import shutil
from mido import MidiFile
from tqdm import tqdm
from music21 import converter

# Your existing data conversion function
def multi_readed(midi_file_pointer):
    midi_file, pointer = midi_file_pointer
    try:
        midi = converter.parse(midi_file)
        return midi
    except Exception as e:
        print(f"Could not convert {midi_file} due to: {e}")
        return None
    
def multi_readed_mido(midi_file_pointer):
    midi_file, pointer = midi_file_pointer
    try:
        midi = MidiFile(midi_file)
        return midi
    except Exception as e:
        print(f"Could not convert {midi_file} due to: {e}")
        return None

def write_progress_file(file_paths, progress_file):
    with open(progress_file, 'w') as f:
        for file in file_paths:
            f.write(file + '\n')

def read_progress_file(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            file_paths = [line.strip() for line in f.readlines()]
    else:
        file_paths = []
    return file_paths

def update_progress_file(file, progress_file):
    file_paths = read_progress_file(progress_file)
    if file in file_paths:
        file_paths.remove(file)
    write_progress_file(file_paths, progress_file)

# Function to handle multiprocessing with timeout and move processed files
def multiprocess_wr_data_conversion(file_paths, num_processes, progress_file, output_folder, timeout=120):
    converted_data_chunks = []
    failed_files = []

    os.makedirs(output_folder, exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        future_to_file = {executor.submit(multi_readed, (file, idx)): file for idx, file in enumerate(file_paths)}
        total_files = len(file_paths)
        processed_files = 0

        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result(timeout=timeout)
                if result is not None:
                    converted_data_chunks.append(result)
                    update_progress_file(file, progress_file)
                    shutil.move(file, os.path.join(output_folder, os.path.basename(file)))
                else:
                    failed_files.append(file)
            except concurrent.futures.TimeoutError:
                print(f"Processing {file} exceeded {timeout} seconds and was terminated.")
                failed_files.append(file)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                failed_files.append(file)
            
            processed_files += 1
            print(f"Processed {processed_files}/{total_files} files")

    return converted_data_chunks, failed_files


# Function to handle multiprocessing with timeout and move processed files
def multiprocess_data_conversion(file_paths, num_processes, timeout=120):
    converted_data_chunks = []
    failed_files = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        future_to_file = {executor.submit(multi_readed_mido, (file, idx)): file for idx, file in enumerate(file_paths)}
        total_files = len(file_paths)
        processed_files = 0

        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result(timeout=timeout)
                if result is not None:
                    converted_data_chunks.append(result)
                else:
                    failed_files.append(file)
            except concurrent.futures.TimeoutError:
                print(f"Processing {file} exceeded {timeout} seconds and was terminated.")
                failed_files.append(file)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                failed_files.append(file)
            
            processed_files += 1
            print(f"Processed {processed_files}/{total_files} files")

    return converted_data_chunks, failed_files