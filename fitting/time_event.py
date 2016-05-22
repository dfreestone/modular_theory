#!/usr/bin/env python
# coding=utf-8
"""
 Name   : time_event.py
 Author : David M. Freestone
 Date   : 03.22.2016

 This time_event module is covered under the modified BSD license
 Copyright (c) 2016, David M. Freestone
 All rights reserved.
 """
import pandas as pd
import numpy as np
from glob import glob
from os.path import basename, join, splitext, expanduser
from datetime import datetime
from shutil import move

from numpy import zeros

from numba import jit, float64, int64

# TODO(David): Do we want a notion of "settings" for different file types?
#   If so, need a way to get/return the settings for a particular file type
#   and a way to update these settings (see file_formats)

# TODO(David): What kind of file_format specific stuff should we have?


def load_file(f, names=None, settings=None):
    """Load a single file into a DataFrame"""
    if names is None and settings is not None:
        names = names_from_settings(settings)
    df = pd.read_csv(f, engine="c", header=0, index_col=False,
                     names=names)
    df["filename"] = basename(f)
    return df


def load_files(files, names=None):
    """Load a list of files into a DataFrame"""
    __load_file__ = lambda f: load_file(f, names=names)
    return pd.concat(list(map(__load_file__, files)))


def load_directory(path, ext, names=None):
    """Load a directory of files into a DataFrame"""
    ext = "*"+ext if "." in ext else "*."+ext
    files = glob(join(path, ext))
    return load_files(files, names)


# @jit(nopython=True)
def cumulative_changes(x):
    return changes(x).cumsum()


# @jit(nopython=True)
def changes(x):
    x = np.hstack((0, x))
    return x[:-1] != x[1:]


@jit(float64[:, :](float64[:], float64, float64, int64, float64), nopython=True)
def mpc_to_tec(rawdata, resolution=0.005, event_tol=0.0001,
               max_event=62, max_time=36000):
    """Convert mpc format (time.event) to [time, event]
        Note: This version is 130x faster than the previous pandas version
    """
    # TODO(David): Write this
    nrows = rawdata.shape[0]
    time_event = np.zeros((nrows, 3))
    max_previous_time = 0
    for idx, datum in enumerate(rawdata):
        time = np.floor(datum)
        event = 1000 * (datum - time)

        if (event <= 0) or (event > max_event):
            continue

        time *= resolution
        # TODO(David): Implement sorting?
        #               Maybe if this is fast enough, just do a gruopby?
        #               Then we can uncomment the line below
        if (time < 0) or (time > max_time): # or (time < max_previous_time):
            continue
        max_previous_time = time

        # The event has to be close enough to an integer to be considered real
        if abs(event-np.round(event)) >= event_tol:
            continue

        time_event[idx, 0] = True
        time_event[idx, 1] = time
        time_event[idx, 2] = np.round(event)
    return time_event


@jit(int64[:](int64[:], int64[:]), nopython=True)
def trialdef(events, pattern):
    """Return the trial number
        Searches for 'pattern' using a simple Finite State Machine
        and returns the trial number (or 0) for every event
    """
    if (pattern[0] < 0) or (pattern[-1] < 0):
        raise ValueError("Must start and end with an event")

    event_index = 0
    event_count = events.size - 1

    pattern_index = 0
    pattern_count = pattern.size - 1

    remove_index = -1
    remove_code = zeros(1 + pattern_count)

    trial_number = 0
    trial = zeros(1 + event_count, dtype=np.int64)

    start_index = 0
    while event_index <= event_count:

        if ((pattern_index == 0) and
                (events[event_index] == pattern[0])):

            pattern_index += 1
            remove_index = -1
            start_index = event_index

        elif ((pattern_index == pattern_count) and
              (events[event_index] == pattern[pattern_count])):
            pattern_index = 0
            remove_index = -1
            trial_number += 1
            trial[start_index:1 + event_index] = trial_number

        elif events[event_index] == pattern[pattern_index]:
            pattern_index += 1
            remove_index = -1

        elif (pattern_index > 0) and (events[event_index] == pattern[0]):
            pattern_index = 0
            event_index -= 1

        elif remove_index > 0:
            for i in range(1 + remove_index):
                if events[event_index] == remove_code[i]:
                    event_index = start_index
                    break

        if pattern[pattern_index] < 0:
            while pattern[pattern_index] < 0:
                remove_index += 1
                remove_code[remove_index] = abs(pattern[pattern_index])
                pattern_index += 1

        event_index += 1
    return trial


def habitest_load_directory(path, ext):
    names = ["time", "event_type", "event_id",
             "event", "register", "comment"]
    df = load_directory(path, ext, names=names)
    return df

def isChurchMPCFile(f):
    try:
        float(f.split(".")[1])
        return True
    except:
        return False

def mpc_load_directory(path):
    files = glob(join(path, "*"))
    files = list(filter(isChurchMPCFile, files))
    df = load_files(files, names=["rawdata"])
    df = pandas_mpc_to_tec(df)
    return df


# TODO(David): This function is are horribly slow
def pandas_mpc_subject_id(filenames):
    return filenames.str.split(".").str[0].str[-4:].astype(int)

# TODO(David): This function is are horribly slow
def pandas_mpc_session_number(filenames):
    """Return the session number given a Series of file names
        (this is probably the fastest way to do it)
    """
    return filenames.str.split(".").str[1].astype(int)

def pandas_mpc_to_tec(df, resolution=0.005, event_tol=0.0001,
                      max_event=62):
    """Return time and event columns in a DataFrame
        (and drop the rawdata column)
    """
    tec = mpc_to_tec(df.rawdata.as_matrix())
    df = df.drop("rawdata", axis=1)
    df["time"] = tec[:, 1]
    df["event"] = tec[:, 2]
    return df[tec[:, 0] > 0]


def isBadSubjectId(f):
    sub = f.split(".")[0][-4:]
    return int(sub) > 8000

def remove_trailing_zero(f):
    fname, ext = splitext(f)
    return fname[:-1] + ext

def rename_bad_mpc_files(fname):
    if isChurchMPCFile(fname) and isBadSubjectId(fname):
        new_fname = remove_trailing_zero(fname)
        move(fname, new_fname)
        return new_fname

def rename_bad_mpc_files_in_path(path):
    """Written to turn say, 18050 to 1805"""
    files = glob(join(path, "*"))
    files = list(map(rename_bad_mpc_files, files))
    return files

def isBadSessionId(f):
    try:
        ext = float(splitext(f)[1])
        return ext > 0.2
    except:
        return False

def rename_bad_mpc_sessions(fname):
    if isBadSessionId(fname):
        new_fname, ext = splitext(fname)
        ext = str(round(float(ext)/10, 3))
        new_fname += "."+ext.split(".")[1]
        move(fname, new_fname)
        return new_fname

def rename_bad_mpc_sessions_in_path(path):
    """Written to turn 0.41 into 0.041"""
    files = glob(join(path, "*"))
    files = list(map(rename_bad_mpc_sessions, files))
    return files




"""
def remove_exit_codes(df):
    return df[df.event_type != "Exit"]

def remove_input_event(df, event):
    return df[(df.event != event+"On")
              & (df.event != event+"Off")]

# TODO(David): What's the best way to integrate low level commands
#              with high level stuff that knows how to work on dataframes?
#              Should it be something like the seaborn or statsmodels API
#              Where we have to give it the columns? The worry is that it
#              then just becomes unnecessary to use these functions?
#              Unless these literally just becomes a file of useful functions?



def session_timestamp(f):
    # Habitest
    name = f.split("_")
    subject = int(name[0])
    session =datetime(*list(map(int, name[1:4]))).timestamp()
    return subject, session

def eventnames_from_ids(ids, names):
    # TODO(David): Write this
    pass

def ids_from_eventnames(names, ids):
    # TODO(David): Write this
    pass

def subset(df):
    # TODO(David): Write this
    pass

def relative_time(time, events):
    # TODO(David): Write this
    pass

def cumulative_trial(trial):
    # TODO(David): Write this
    pass

def trial_def(events, pattern):
    # TODO(David): Write this
    pass
"""
