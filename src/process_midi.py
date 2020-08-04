import argparse
import os
import re
import numpy as np
from collections import Counter

# pip install git+https://github.com/gnithie/python-midi/
import midi

from constants import *


nsteps = 64
nsteps_per_beat = 2
max_beats = 32
pitch = Counter()

def load_both_data(path):
    count = 0
    X, y1, y2 = [], [], []
    rejected = accepted = 0
    for dir in (os.listdir(path)):
        print(dir.split()[0])
        for root, dirs, files in os.walk(os.path.join(path, dir)):
    # for root, dirs, files in os.walk(path):
            genre_label = [genre for genre in TARGET_GENRES if genre.lower() in root.lower()]
            section_label = [section for section in TARGET_SECTIONS if section.lower() in root.lower()]
            # print(data)
            if len(genre_label) > 0 and len(section_label) > 0:

                for file in files:
                    # print(root, file)
                    if file.endswith('.mid'):
                        filepath = os.path.join(root, file)
                        # print(filepath)
                        midi_data = read_midi(filepath)
                        info = midi_info(midi_data)
                        if (info["time_sig_num"] != 4 or
                            (info["track_length_in_beats"] not in (8, 16, 32))):
                            rejected += 1
                            continue
                        accepted += 1
                        numpy_data = midi2numpy(midi_data)
                        # np.add(numpy_data, TARGET_LABELS.index(dir.split()[0]))
                        X.append(numpy_data)
                        # print(data[0])
                        y1.append(TARGET_GENRES.index(genre_label[len(genre_label) -1]))
                        y2.append(TARGET_SECTIONS.index(section_label[len(section_label) -1]))
    print('rejected', rejected, 'accepted', accepted)
    X = np.array(X)
    y1 = np.array(y1)
    y2 = np.array(y2)
    np.save(path.rstrip("/") + "_both_X.npy", X)
    np.save(path.rstrip("/") + "_both_y1.npy", y1)
    np.save(path.rstrip("/") + "_both_y2.npy", y2)
    print('Result')
    print(X.shape)
    print(y1.shape, y2.shape)
    print(Counter(y1))
    print(Counter(y2))

def load_data(path):
    # for dir in (os.listdir(path)):
    #     print(dir.split()[0])
    count = 0
    X, y = [], []
    rejected = accepted = 0
    for root, dirs, files in os.walk(path):
        data = [genre for genre in TARGET_SECTIONS if genre.lower() in root.lower()]
        # print(data)
        if len(data) > 0:

            for file in files:
                # print(root, file)
                if file.endswith('.mid'):
                    filepath = os.path.join(root, file)
                    # print(filepath)
                    midi_data = read_midi(filepath)
                    info = midi_info(midi_data)
                    if (info["time_sig_num"] != 4 or
                        (info["track_length_in_beats"] not in (8, 16, 32))):
                        rejected += 1
                        continue
                    accepted += 1
                    numpy_data = midi2numpy(midi_data)
                    # np.add(numpy_data, TARGET_LABELS.index(dir.split()[0]))
                    X.append(numpy_data)
                    print(data[0])
                    y.append(TARGET_SECTIONS.index(data[len(data) -1]))
    print('rejected', rejected, 'accepted', accepted)
    X = np.array(X)
    y = np.array(y)
    np.save(path.rstrip("/") + "_X1.npy", X)
    np.save(path.rstrip("/") + "_y1.npy", y)
    print('Result')
    print(X.shape)
    print(y.shape)
    print(Counter(y))

def convert_midi2numpy(path):
    X, y = [], []
    rejected = accepted = 0
    for dir in (os.listdir(path)):
        print(dir.split()[0])
        for root, dirs, files in os.walk(os.path.join(path, dir)):
            for file in files:
                # print(root, file)
                if file.endswith('.mid'):
                    filepath = os.path.join(root, file)
                    # print(filepath)
                    midi_data = read_midi(filepath)
                    info = midi_info(midi_data)
                    if (info["time_sig_num"] != 4 or
                        (info["track_length_in_beats"] not in (8, 16, 32))):
                        rejected += 1
                        continue
                    accepted += 1
                    numpy_data = midi2numpy(midi_data)
                    # np.add(numpy_data, TARGET_LABELS.index(dir.split()[0]))
                    X.append(numpy_data)
                    y.append(TARGET_GENRES.index(dir.split()[0]))
    print('rejected', rejected, 'accepted', accepted)
    X = np.array(X)
    y = np.array(y)
    np.save(path.rstrip("/") + "_X.npy", X)
    np.save(path.rstrip("/") + "_y.npy", y)
    print('Result')
    print(X.shape)
    print(y.shape)
    print(pitch)
    
def midi2numpy(midi_data):
    info = midi_info(midi_data)

    data_len = nsteps_per_beat * info["track_length_in_beats"]
    data = np.zeros((len(DRUM_NAMES), data_len), dtype='int')
    
    for track in midi_data:
        for event in track:
            if event.name == "Note On" and event.get_velocity() > 0:
                pitch[event.get_pitch()] += 1
                try:
                    index = PITCH_DRUMINDEX[event.get_pitch()]
                except:
                    continue
                data[index, convert_time(event.tick, info["track_length_in_ticks"], data_len)] = (event.get_velocity())
    if info["track_length_in_beats"] == 8:
        data = np.concatenate((data, data, data, data), axis=1)
    elif info["track_length_in_beats"] == 16:
        data = np.concatenate((data, data), axis=1)
    else:
        pass
    assert data.shape[1] == nsteps
    return data

def convert_time(ticks, maxticks, nsteps_this_loop):
    x = int(round(nsteps_this_loop * float(ticks) / maxticks))
    if x >= nsteps_this_loop:
        x = nsteps_this_loop -1
    return x

def read_midi(file):
    if not file.endswith('.mid'):
        raise ValueError
    midi_data = midi.read_midifile(file)
    midi_data.make_ticks_abs()
    return midi_data

def midi_info(midi_data):
    info = {}
    info["resolution"] = midi_data.resolution

    for track in midi_data:
        for event in track:
            if event.name == "Track Name":
                info["track_name"] = event.text
            if event.name == "Set Tempo":
                info["bpm"] = event.get_bpm()
                info["mpqn"] = event.get_mpqn()
            if event.name == "Time Signature":
                info["time_sig_num"] = event.get_numerator()
                info["time_sig_denom"] = event.get_denominator()
            if event.name == "End of Track":
                if True: 
                    maxticks = event.tick
                    info["track_end_in_ticks"] = event.tick
    info["track_length_in_beats"] = int(round(maxticks / float(midi_data.resolution)))
    info["track_length_in_ticks"] =  info["track_length_in_beats"] * midi_data.resolution
    return info

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--midipath', 
        type = str, 
        default = GET_DEFAULTS["midi_path_full"],
        help = 'path to midi folder, default ' + GET_DEFAULTS["midi_path_full"]
    )
    params = parser.parse_args()

    # convert_midi2numpy(params.midipath)
    # load_data(params.midipath)
    load_both_data(params.midipath)


