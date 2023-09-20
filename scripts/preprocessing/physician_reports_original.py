# %%
import numpy as np
import pandas as pd
import logging

# %%
from glob import glob
import re

def natural_key(string):
    """ provides a human-like sorting key of a string """
    p = r'(\d+)'
    key = [int(t) if t.isdigit() else None for t in re.split(p, string)]
    return key

# check whether this can be replaced by natural key
def _session_key(string):
    """ sort the file name by session """
    p = r'(s\d*)_'
    return re.findall(p, string)


def _time_key(file_name):
    """ provides a time-based sorting key """
    # the splits are specific to tuh abnormal eeg data set
    splits = file_name.split('/')
    p = r'(\d{4}_\d{2}_\d{2})'
    [date] = re.findall(p, splits[-2])
    date_id = [int(token) for token in date.split('_')]
    recording_id = natural_key(splits[-1])
    session_id = _session_key(splits[-2])
    return date_id + session_id + recording_id


def read_all_file_names(path, extension, key="time"):
    """ read all files with specified extension from given path
    :param path: parent directory holding the files directly or in
    subdirectories
    :param extension: the type of the file, e.g. '.txt' or '.edf'
    :param key: the sorting of the files. natural e.g. 1, 2, 12, 21
        (machine 1, 12, 2, 21) or by time since this is
        important for cv. time is specified in the edf file names
    """
    assert key in ["natural", "time"], "unknown sorting key"
    file_paths = glob(path + '**/*' + extension, recursive=True)
    if key == "time":
        sorting_key = _time_key
    else:
        sorting_key = natural_key
    file_names = sorted(file_paths, key=sorting_key)

    assert len(file_names) > 0, ("something went wrong. Found no {} files in {}"
                                 .format(extension, path))
    return file_names

# %%
def uniqe_ids(path):
    all_reports = np.array(read_all_file_names(path, extension=".txt", key="time"))
    ids = [r.split("/")[-3] for r in all_reports]
    return all_reports, np.unique(ids)

# %%
all_reports, unique_patient_ids = uniqe_ids("/data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/")

# %%
len(all_reports)

# %%
len(unique_patient_ids)

# %%
train_normal_reports, train_normal_unique_patient_ids = uniqe_ids("/data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/eval/")

# %%
len(train_normal_reports), len(train_normal_unique_patient_ids)

# %%
train_abnormal_reports, train_abnormal_unique_patient_ids = uniqe_ids("/data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/")

# %%
len(train_abnormal_reports), len(train_abnormal_unique_patient_ids)

# %%
len(np.intersect1d(train_normal_unique_patient_ids, train_abnormal_unique_patient_ids))

# %%


# %%


# %%


# %%


# %%
def remove_exccessive_white_space(string):
    replaces = re.sub(r'\s+', ' ', string)
    return replaces.strip()

# %%
def merge_columns(df, merge_to, merge_from, drop=True):
    # TODO: check that no data is overwritten by merging?
    assert merge_to in df.columns, "column {} not found in dataframe".format(merge_to)
    assert merge_from in df.columns, "column {} not found in dataframe".format(merge_from)
    df[merge_to][pd.isna(df[merge_to])] = df[merge_from][pd.isna(df[merge_to])]
    if drop:
        df = df.drop(merge_from, axis=1)
    return df

# %%
def merge_several_columns(df, merge_to, merge_from_several, drop=True):
    for column in merge_from_several:
        df = merge_columns(df, merge_to, column, drop=drop)
    return df

# %%
# assume sections are given in all caps seperated only by white space characters and followed by a colon
# assume that section text starts with a colon and is everything in between two sections
categoy_pattern = r"^([A-Z\s]{2,}):{1}"
content_pattern = r":(.*)"

# %%
i_start = 998
i_stop = 1001
df = pd.DataFrame()
for report in all_reports[i_start:i_stop]:
    with open(report, "rb") as f:
        content = f.readlines()
    assert content, "error reading {}".format(report)
    content = b'\n'.join([line.strip() for line in content]).strip().decode("latin-1")
    categories = re.findall(categoy_pattern, content, re.MULTILINE)   
    assert len(categories) > 0, "no categories found"

    splits = report.split('/')
    # add subject id, session id, label, data path and set information
    df_row = {"SUBJECT": splits[-3], "SESSION": splits[-2], "LABEL": splits[-6], 
              "PATH": report, "SET": splits[-7]}
    
    # some recordings have multiple entries per category. skip these files for now, add later manually
    if len(np.unique(categories)) != len(categories):
        df = df.append(df_row, ignore_index=True)
        continue
    
    # go through all subsequent pairs of categories, extract text inbetween and assign it to start category
    for j in range(len(categories) - 1):
        start = categories[j]
        stop = categories[j + 1]
        match = re.findall(start + content_pattern + stop, content, re.DOTALL)
        assert len(match) == 1, "found more than one match!"
        # remove multiple spaces and newlines
        start = ' '.join(start.split())
        df_row.update({start: remove_exccessive_white_space(match[0])})
        
    # take all text that appears after last category and assign 
    match = re.findall(stop + content_pattern, content, re.DOTALL)
    assert len(match) == 1, "found more than one match!"
    # remove multiple spaces and newlines
    stop = ' '.join(stop.split())
    df_row.update({stop: remove_exccessive_white_space(match[0])})
    df = df.append(df_row, ignore_index=True)

# %%
len(df)

# %%
df.head(3)

# %%
df = merge_several_columns(df, "CLINICAL CORRELATION", ["CORRELATION", "CLINICAL COURSE", "CLINICAL CORRELATIONS", "CLINICAL CORRELATE",
                                                        "CLINICAL INTERPRETATION", "CLINICAL CORR ELATION", "NOTE"])  # not sure about merging note

# %%
df = merge_several_columns(df, "CLINICAL HISTORY", ["HISTORY", "CLINICAL", "M CLINICAL HISTORY", "EEG REPORT CLINICAL HISTORY", 
                                                    "HOSPITAL COURSE", "BASELINE EEG CLINICAL HISTORY", "EEG NUMBER",
                                                    "ORIGINAL CLINICAL HISTORY"])

# %%
df = merge_several_columns(df, "DESCRIPTION OF THE RECORD", ["DESCRIPTION OF RECORD", "DESCRIPTION RECORD", "DESCRIPTION OF RECORDING", 
                                                             "DESCRIPTION OF THE RECORDING", "DESCRIPTION OF THE PROCEDURE", 
                                                             "DESCRIPTION OF BACKGROUND", "DESCRIPTION OF PROCEDURE", "OF THE RECORD",
                                                             "DESCRIPTION THE RECORD"])

# %%
df = merge_several_columns(df, "MEDICATIONS", ["MEDICATION", "CURRENT MEDICATIONS", "MEDICINES"])

# %%
df = merge_several_columns(df, "HEART RATE", ["HEAR RATE", "HR"])

# %%
df = merge_several_columns(df, "IMPRESSION", ["CLINICAL IMPRESSION"])

# %%
df = merge_several_columns(df, "FINDINGS", ["ABNORMAL FINDINGS"])

# %%
df = merge_several_columns(df, "EVENTS", ['SEIZURE EVENTS', 'SEIZURES OR EPISODES', 'EVENT', 'EPISODES', 'CLINICAL EVENTS',
                                          "EPISODES OR EVENTS", "EPISODES DURING THE RECORDING", "REFERRING FOR STUDY",
                                          "EVENTS OF PUSHBUTTON", "SEIZURES", "SEIZURE ACTIVITY"])

# %%
df = merge_several_columns(df, "TECHNICAL DIFFICULTIES", ["TECHNICAL PROBLEMS", "TECHNICAL DIFFICULTY", "CLINICAL DIFFICULTIES", "TECHNICAL DISCHARGES", 
                                                          "TECHNICAL NOTES", "TECHNICAL ISSUES", "TECHNIQUE DIFFICULTIES", "TECHNICAL CONSIDERATIONS",
                                                          "TECHNICAL QUALITY", "TECHNICAL", "ARTIFACTS"])

# %%
df = merge_several_columns(df, "CONDITION OF THE RECORDING", ["CONDITIONS OF THE RECORDING", "CONDITION OF RECORDING", 
                                                              "CONDITIONS OF RECORDING"])

# %%
df = merge_several_columns(df, "REASON FOR STUDY", ["REASON", "REASON FOR STUDIES", "REASON FOR EGG", "REASON FOR THE STUDY"])

# %%
df = merge_several_columns(df, "FINDINGS", ["DIAGNOSES", "DIAGNOSIS", "ABNORMAL DISCHARGES", "ABNORMAL DISCHARGE", 
                                            "EEG", "RECOMMENDATIONS"])  # not sure about merging recommendations

# %%
df = merge_several_columns(df, "PAST MEDICAL HISTORY", ["PAST HISTORY"])

# %%
df = merge_several_columns(df, "ACTIVATION PROCEDURES", ["ACTIVATION PROCEDURE", "ACTIVATING PROCEDURES", ])

# %%
df = merge_several_columns(df, "REASON FOR STUDY", ["REASON FOR EEG", "REASON FOR PROCEDURE"])

# %%
for drop_column in ["RECORDING TIMES", "RECORDING START TIME", "RECORDING END TIME", "RECORD FINISH TIME", "RECORD START TIME", 
                    "TOTAL LENGTH OF THE RECORDING", "RECORDING LENGTH", "TIME OF RECORDING", "LENGTH OF ELECTROENCEPHALOGRAM", 
                    "EEG LENGTH", "LENGTH OF EEG", "LENGTH OF PROCEDURE", "LENGTH OF THE RECORDING", "LENGTH OF THE EEG", 
                    "LENGTH OF RECORDING", "STUDY DATE", "DATE OF RECORDING", "EGG LENGTH", "TIME", "DURATION OF STUDY", 
                    "STUDY DURATION", "DATE OF THE RECORDING", "DATE OF STUDY", "DATES OF STUDY", "DT", "DD", "DENTAL PROBLEMS", 
                    "STAGES", "REASON FOR SEIZURES", "SEIZURES OR PUSHBUTTON EVENTS", "FEATURES", "INPATIENT ROOM", "EKG",
                    "DATE", "SLEEP"]:
    df = df.drop(drop_column, axis=1)

# %%
df.head(3)

# %%
print([c for c in df.columns])

# %%


# %%
len(df) - pd.isna(df).sum()

# %%


# %%


# %% [markdown]
# checking individual cases

# %%
column = "INTRODUCTION"

# %%
df[column][~pd.isna(df[column])]

# %%
df.iloc[1943]

# %%
df["CLINICAL COURSE"][~pd.isna(df["CLINICAL COURSE"])]

# %%
df["TECHNICAL DISCHARGES"][~pd.isna(df["TECHNICAL DISCHARGES"])]

# %%
df["ABNORMAL DISCHARGE"][~pd.isna(df["ABNORMAL DISCHARGE"])]

# %%
df["CLINICAL DIFFICULTIES"][~pd.isna(df["CLINICAL DIFFICULTIES"])]

# %%
df["PATH"].iloc[299]

# %%
!cat /data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/normal/01_tcp_ar/083/00008366/s001_2011_10_24/00008366_s001.txt

# %%
!cat /data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/normal/01_tcp_ar/067/00006766/s001_2010_07_06/00006766_s001.txt

# %% [markdown]
# df.to_csv("/data/schirrmr/gemeinl/tuh-abnormal-eeg/reports/reports_{}_{}.csv".format(i_start, i_stop))

# %%
!cat /data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/normal/01_tcp_ar/103/00010316/s001_2013_05_14/00010316_s001.txt

# %%
!cat /data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/abnormal/01_tcp_ar/100/00010026/s001_2013_02_19/00010026_s001.txt

# %%
!cat /data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/abnormal/01_tcp_ar/022/00002253/s003_2011_05_12/00002253_s003.txt

# %%
!cat /data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/abnormal/01_tcp_ar/000/00000068/s010_2013_01_21/00000068_s010.txt

# %%
!cat /data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/abnormal/01_tcp_ar/085/00008512/s001_2012_01_09/00008512_s001.txt

# %%
!cat /data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/abnormal/01_tcp_ar/085/00008512/s001_2012_01_09/00008512_s001.txt

# %%
!cat /data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/abnormal/01_tcp_ar/082/00008251/s001_2011_11_26/00008251_s001.txt

# %%
!cat /data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/normal/01_tcp_ar/083/00008337/s001_2011_06_30/00008337_s001.txt

# %%
!cat /data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/abnormal/01_tcp_ar/065/00006535/s005_2012_07_18/00006535_s005.txt

# %%
!cat /data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/abnormal/01_tcp_ar/106/00010639/s001_2013_08_22/00010639_s001.txt

# %%
!cat /data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/normal/01_tcp_ar/089/00008913/s001_2012_07_12/00008913_s001.txt

# %%
!cat /data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/abnormal/01_tcp_ar/092/00009231/s002_2012_10_06/00009231_s002.txt

# %%
!cat /data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/eval/abnormal/01_tcp_ar/100/00010003/s001_2013_02_18/00010003_s001.txt

# %%
!cat /data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/abnormal/01_tcp_ar/092/00009231/s002_2012_10_06/00009231_s002.txt

# %%
re.findall(categoy_pattern, "INTRODUCTION:  Digital video EEG is performed at the bedside using standard 10-20 system of electrode placement with 1 channel of EKG.  Hyperventilation and photic stimulation are performed."
"October 6-7\n"
"DESCRIPTION OF THE RECORD:\n"
"As the tracing began, there was a great deal of muscle artifact in the EEG.  In addition to the eye blink artifact, there", re.MULTILINE)

# %%
!cat /data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/abnormal/01_tcp_ar/092/00009213/s002_2012_09_17/00009213_s002.txt

# %%
!cat /data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/abnormal/01_tcp_ar/065/00006535/s005_2012_07_18/00006535_s005.txt

# %%
!cat /data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/abnormal/01_tcp_ar/095/00009578/s003_2013_03_01/00009578_s003.txt

# %%
!cat /data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/normal/01_tcp_ar/098/00009896/s001_2013_04_24/00009896_s001.txt

# %%
df.iloc[2561]

# %%



