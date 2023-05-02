# read a text file
path = "/data/datasets/TUH/EEG/tuh_eeg_abnormal/v2.0.0/"
file_name = '_AAREADME.txt'

with open(path + file_name, 'r') as file:
    data = file.read().replace('\n', '')
    print(data)