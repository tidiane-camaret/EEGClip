# EEGClip : Learning EEG representations using natural language descriptions


## Running instructions


### 1. Contrastive training on the TUH EEG Abnormal Corpus

Trains the EEGClipModel, defined in ```EEGClip/clip_models.py```, on the TUH EEG Abnormal Corpus (first 70% of the subjects for training) 

```python3 scripts/clip/clip_tuh_eeg.py```

Documentation for each parameter can be found in the script.
Note : you will need to modify the path to the TUH EEG Abnormal Corpus in the script.

### 2. Evaluation on the TUH EEG Abnormal Corpus

2.1. Few shot classification

Trains and evaluates a classification model on a given task.

```python3 scripts/clip/clip_tuh_eeg.py```

```