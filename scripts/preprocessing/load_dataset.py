"""
Shows how to load a saved dataset

"""

from braindecode.datautil import load_concat_dataset
from braindecode.preprocessing import create_windows_from_events

DATAPATH = "data/TUH/TUHMock"

dataset_loaded = load_concat_dataset(
    path=DATAPATH,
    preload=True,
    #ids_to_load=[1, 3],
    target_name=None,
)

dataset_loaded.description

"""
windows_dataset = create_windows_from_events(
    concat_ds=dataset_loaded,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
)

windows_dataset.description
"""