#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from glob import glob
from os.path import basename, isfile, join
from time_event import trialdef, cumulative_changes


def session_info(f):
    f = basename(f).split(".")[0]
    subject = int(f[3:6])
    session = int(f[6:])
    return subject, session


def read_session(f):
    df = pd.read_csv(f, sep="\t", engine="c",
                     header=None, names=["time", "event"])
    df["subject"], df["session"] = session_info(f)
    return df


def gtrialdef(events, pattern):
    return trialdef(events.as_matrix(), np.array(pattern))


def gcumulative_changes(trial):
    idx = trial > 0
    trial[idx] = cumulative_changes(trial[idx])
    return trial


def relative_time(t):
    return t - t.iloc[0]


def create_modular_theory_file(how="acquisition", overwrite=False):
    path = "../data/modular_theory2007"
    filename = "Data_Modular_Theory2007.csv"
    if overwrite or not isfile(join(path, filename)):
        files = glob(join(path, "raw_data", "*.txt"))

        print("Loading files...", end="")
        df = pd.concat([read_session(f) for f in files])
        print("done.")

        print("Assigning conditions...", end="")
        assign_phase(df, 1, 30, "acquisition")
        assign_phase(df, 31, 65, "extinction")
        assign_phase(df, 66, 75, "reacquisition")
        assign_phase(df, 76, 120, "repeated_extinction")
        assign_extinction_conditions(df, [437, 440, 444, 445], "FI30")
        assign_extinction_conditions(df, [434, 435, 439, 443], "FI60")
        assign_extinction_conditions(df, [433, 438, 441, 442], "FI120")

        df["phase"] = df.phase.astype("category")
        df["extinction_condition"] = df.extinction_condition.astype("category")
        print("done.")

        if how == "acquisition":
            df = df[df.phase=="acquisition"]

        print("Defining trials...", end="")
        trial_types = {"FI30": 50, "FI60": 51, "FI120": 52}
        for trialname in trial_types:
            startcode = trial_types[trialname]
            df["trial_"+trialname] = df.groupby(["subject", "session"]).event.transform(gtrialdef, [startcode, 53])
            df["trial_"+trialname] = df.groupby("subject")["trial_"+trialname].transform(gcumulative_changes)
            df["time_"+trialname] = df.groupby(["subject", "trial_"+trialname]).time.transform(relative_time)
        print("done.")

        print("Saving...", end="")
        df.to_csv(join(path, filename), index=False)
        print("done.")
    else:
        print("Creating file...done (already present)")


def assign_phase(df, start, stop, name):
    if "phase" not in df.columns:
        df["phase"] = ""
    df.ix[(df.session >= start) & (df.session <= stop), "phase"] = name


def assign_extinction_conditions(df, subjects, name):
    if "extinction_condition" not in df.columns:
        df["extinction_condition"] = ""
    df.ix[df.subject.isin(subjects), "extinction_condition"] = name

if __name__ == "__main__":
    create_modular_theory_file()
