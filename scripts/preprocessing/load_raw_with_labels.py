
import glob
import pandas as pd

from tueg_stardard_preproc import standard_preproc

tueg_path = '/home/jovyan/mne_data/TUH/tuh_eeg_abnormal/'
out_path = '/home/jovyan/mne_data/TUH_PRE/tuh_eeg_abnormal_clip/'
all_edf_paths = glob.glob(tueg_path + '**/*.edf', recursive=True)



error_log = []
for edf_in_path in all_edf_paths:
    print("processing " + edf_in_path)
    try:
        standard_preproc(
            edf_in_path=edf_in_path,
            edf_out_path=edf_in_path.replace(tueg_path, out_path),
            min_mins=-1,
            max_hours=-1,
        )
    except Exception as e:
        error_log.append((edf_in_path, e))
    else:
        error_log.append((edf_in_path, 'success'))

log_df = pd.DataFrame(error_log, columns=['rec', 'reason'])
log_df.to_csv(out_path + 'preprocessing_log.csv')




