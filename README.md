# EEGClip : Learning EEG representations using natural language descriptions


## Running instructions


1. Contrastive training on the TUH EEG Abnormal Corpus

Trains the EEGClipModel, defined in ```EEGClip/clip_models.py```, on the TUH EEG Abnormal Corpus. The model is trained using the contrastive loss defined in ```models/contrastive.py```. The model is trained for 100 epochs, with a batch size of 32, and a learning rate of 1e-4. The model is saved in ```models/clip_tuh_eeg.pth```. The model is trained on a single GPU.

```python3 scripts/clip/clip_tuh_eeg.py```

```

2. Contrastive pretraining on the TUH EEG Abnormal Corpus

```python3 EEGClip/scripts/clip/clip_tuh_eeg.py```

```