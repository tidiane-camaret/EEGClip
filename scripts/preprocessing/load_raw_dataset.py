import numpy as np
import matplotlib.pyplot as plt
import mne

from braindecode.datasets import TUH
from braindecode.preprocessing import (
    preprocess, Preprocessor, create_fixed_length_windows, create_windows_from_events, scale as multiply)

mne.set_log_level('ERROR')  # avoid messages everytime a window is extracted

TUH_PATH = '/home/jovyan/mne_data/TUH/tuh_eeg_abnormal/v2.0.0'
N_JOBS = 8  # specify the number of jobs for loading and windowing
tuh = TUH(
    path=TUH_PATH,
    recording_ids=[:30],
    target_name=('gender'),
    preload=False,
    add_physician_reports=True,
    n_jobs=1 if TUH.__name__ == '_TUHMock' else N_JOBS,  # Mock dataset can't
    # be loaded in parallel
)

print("length of dataset : ", len(tuh))

#show last example 
x, y = tuh[-1]
print('x:', x)
print('y:', y)

# save description of the dataset
print("columns : ", list(tuh.description.columns))
print(tuh.description)

window_size_samples = 1000
window_stride_samples = 1000
tuh_windows = create_fixed_length_windows(
    tuh,
    window_size_samples=window_size_samples,
    window_stride_samples=window_stride_samples,
    drop_last_window=False,
    n_jobs=N_JOBS,
)

#for x, y, ind in tuh_windows:
#    print("Window shape: {}, target: {}, ind: {}".format(x.shape, y, ind))
