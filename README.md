# EEGClip : Learning EEG representations using natural language descriptions

## Running instructions

*Please install the required dependecies using ```pip install -r requirements.txt```, as well as the local package with ```pip install -e .```*
*Note : you may need to modify the path to the datasets and to the pretrained models in ```EEGClip_config.py```*

*Detailed documentation for each parameter can be found in the respective scripts.*


### 1. Contrastive training on the TUH EEG Abnormal Corpus

Trains the EEGClipModel, defined in ```EEGClip/clip_models.py```, on the TUH EEG Abnormal Corpus (first 70% of the subjects for training) 

```python3 scripts/clip/train_eegclip_tuh.py```


### 2. Evaluation on the TUH EEG Abnormal Corpus


#### 2.1. Label decoding

Trains and evaluates a classifier model on a given task (pathological, age, gender ...) using the frozen EEG encoder trained in step 1. (first 70% of the subjects for training, last 30% for testing)

```python3 scripts/classif/classification_tuh.py```

It is possible to reduce the training set **(few-shot decoding)** by specifing the **--train_frac** parameter (by how much the training set should be divided, eg 5 for 20% of the training set).

It is also possible to change the encoder to a fully trainable one using the **--weights** and **--freeze_encoder** parameters. 

#### 2.2. Zero-shot label decoding

Evaluates zero-shot accuracy of the EEG encoder on a given task (pathological, age, gender ...)
For a given recording, we first extract its representations, and measure its distance to anchor sentences ("This is a normal recording", etc..)

```python3 scripts/classif/classification_zero_shot_tuh.py```

