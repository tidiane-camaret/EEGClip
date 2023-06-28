# EEGClip : Learning EEG representations using natural language descriptions


## Running instructions


### 1. Contrastive training on the TUH EEG Abnormal Corpus

Trains the EEGClipModel, defined in ```EEGClip/clip_models.py```, on the TUH EEG Abnormal Corpus (first 70% of the subjects for training) 

```python3 scripts/clip/train_eegclip_tuh.py```

Documentation for each parameter can be found in the script.


### 2. Evaluation on the TUH EEG Abnormal Corpus

2.1. label decoding

Trains and evaluates a classifier model on a given task (pathological, age, gender ...) using the frozen EEG encoder trained in step 1. (first 70% of the subjects for training, last 30% for testing)

```python3 scripts/classif/classification_tuh.py```

It is possible to reduce the training set **(few-shot decoding)** by specifing the **--train_frac** parameter (number between 0 and 1).

It is also possible to change the encoder to a fully trainable one using the **--weights** and **--freeze_encoder** parameters. 
Documentation for each parameter can be found in the script.

Note : you will need to modify the path to the TUH EEG Abnormal Corpus and of the pretrained model in the script.

2.2. zero-shot label decoding

Evaluates zero-shot accuracy of the EEG encoder on a given task (pathological, age, gender ...)
For a given recording, we first extract its representations, and measure its distance to anchor sentences ("This is a normal recording", etc..)

```python3 scripts/classif/classification_zero_shot_tuh.py```

Documentation for each parameter can be found in the script.

Note : you will need to modify the path to the TUH EEG Abnormal Corpus and of the pretrained model in the script.
