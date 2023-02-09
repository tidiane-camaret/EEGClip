import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import mne

from braindecode.datasets.tuh import _parse_age_and_gender_from_edf_header


ar_ch_names = sorted([
    'EEG A1-REF', 'EEG A2-REF',
    'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
    'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
    'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
    'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'])
le_ch_names = sorted([
    'EEG A1-LE', 'EEG A2-LE',
    'EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE',
    'EEG C4-LE', 'EEG P3-LE', 'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE',
    'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE',
    'EEG T6-LE', 'EEG FZ-LE', 'EEG CZ-LE', 'EEG PZ-LE'])
sfreq = 100


class TooLongRecError(Exception):
    pass


class TooShortRecError(Exception):
    pass


class ExportFailedError(Exception):
    pass


def preprocess(file_path, min_mins, max_hours):
    # read edf
    raw = mne.io.read_raw_edf(file_path, verbose='error')
    
    # drop rec if it is shorter than 2 min
    if min_mins != -1:
        if raw.n_times / raw.info['sfreq'] < min_mins * 60:
            raise TooShortRecError(
                f'Rec shorter than {min_mins} minutes. '
                'Skipping.'
            )

    # if recs are longer given hours, fail
    if max_hours != -1:
        if raw.n_times > raw.info['sfreq'] * 60 * 60 * max_hours:
             raise TooLongRecError(
                 f'Rec longer than {max_hours} hours. '
                 'Skipping to avoid out of memory error.'
             )
    
    # pick 21 channels
    if len(np.intersect1d(ar_ch_names, raw.ch_names)) == len(ar_ch_names):
        raw = raw.pick_channels(ar_ch_names)
    elif len(np.intersect1d(le_ch_names, raw.ch_names)) == len(le_ch_names):
        raw = raw.pick_channels(le_ch_names)
    else:
        raise ValueError(f'Desired channels not found {raw.ch_names}.')

    # make sure unit for channels is volts
    if not all(ch['unit'] == mne.io.constants.FIFF.FIFF_UNIT_V for ch in raw.info['chs']):
        return ValueError(f'Incorrect data unit discovered {raw.info["chs"]}.')

    # clip to -800/+800 microvolt
    raw.load_data()
    raw = raw.apply_function(lambda x: np.clip(x, -800/1e6, 800/1e6))

    # apply common average referencing
    raw = raw.set_eeg_reference(ref_channels='average')

    # resample to 100 hz
    raw = raw.resample(sfreq)
    
    # remove measurement date as sometimes files cannot be saved because of it
    raw = raw.set_meas_date(None)

    # rename channels
    mapping = {
        ch_name: ch_name.replace('-REF', '').replace('-LE', '').replace('EEG ','') 
        for ch_name in raw.ch_names
    }
    raw = raw.rename_channels(mapping)
    
    # raw.info['subject_info'] is partially written to the edf header,
    # so fake the entries, such that the patient age and gender 
    # is preserved and can still be parsed with braindecode
    # TUH dataset classes
    age, gender = _parse_age_and_gender_from_edf_header(file_path)
    gender_map = {'F': 0, 'M': 1, 'X': 2}
    raw.info['subject_info'] = {
        # y, m, d
        #'birthday': (birth_year, 1, 1), 
        # 0: F, 1: M, 2: X
        'sex': gender_map[gender], 
        # <first_name>_<last_name>
        # instead of the name of the subject, which we don't know,
        # use the subject id
        # 'last_name': subj_id,
        # abuse the 'hand' field to add the age in the form 
        # 'Age:<age>' as this is what the braindecode classes parse
        # since this is how it was written in the original tuh file headers
        'hand': ' '.join(['None', f'Age:{age}']),
    }
    return raw


def save(raw, mean_n_std, file_path):
    # create the output file name by replacing data path with output path
    # so subdirectory structure remains the same
    #out_file = file_path.replace(data_path, out_path)
    this_out_path = os.path.dirname(file_path)
    if not os.path.exists(this_out_path):
        os.makedirs(this_out_path)
    # write the file
    try:
        mne.export.export_raw(file_path, raw, add_ch_type=True, overwrite=True)
    except:
        # some rec failed with 'RuntimeError: writeSamples() for channelFP1 returned error: -25'
        # but created an edf file on disk anyways that was corrupted
        # so if this happens, delete the file if it exists
        raise ExportFailedError('MNE export failed. Deleting artifacts.')
        if os.path.exists(file_path):
            os.remove(file_path)
    else:
        mean_n_std.to_csv(file_path.replace('.edf', '_stats.csv'))


def standard_preproc(edf_in_path, edf_out_path, min_mins, max_hours):
    raw = preprocess(edf_in_path, min_mins, max_hours)
    # compute mean and std per ch in microvolts
    means = raw.get_data(units='uV').mean(axis=1)
    stds = raw.get_data(units='uV').std(axis=1)
    # store mean and std to file
    mean_n_std = pd.DataFrame(zip(means, stds), index=raw.ch_names, columns=['mean', 'std'])
    save(raw, mean_n_std, edf_out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # args for decoding
    parser.add_argument('--edf-in-path', type=str)
    parser.add_argument('--edf-out-path', type=str)
    parser.add_argument('--max-hours', type=int)
    parser.add_argument('--min-mins', type=int)
    # args for storing run details
    parser.add_argument('--run-name', type=str)
    args, unknown = parser.parse_known_args()
    args = vars(args)
    print(datetime.now())
    print(pd.Series(args))
    if unknown:
        raise ValueError(f'There are unknown input parameters: {unknown}')

    run_name = args.pop('run_name')
    # run the actual code
    standard_preproc(**args)
