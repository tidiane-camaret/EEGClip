import os
import pandas as pd


# reports were already painstakingly processed by Gemeinl
# see https://github.com/gemeinl/auto-eeg-diagnosis-comparison/blob/master/code/physician_reports.ipynb

current_dir = os.getcwd() + '/scripts/text_preprocessing/'
# dataframe created by the TUH class
dataset_df=pd.read_csv(current_dir + '/tuh_description.csv')
# processed dataframe
processed_df = pd.read_csv(current_dir + 'tuh_description_processed.csv')

# add common column to both dataframes
dataset_df['new_col'] = 's' + dataset_df['session'].astype(str).str.pad(3, fillchar='0')  + '_' + dataset_df['year'].astype(str) + '_' + dataset_df['month'].astype(str).str.pad(2, fillchar='0') + '_' + dataset_df['day'].astype(str).str.pad(2, fillchar='0') + dataset_df["subject"].astype(str)
processed_df['new_col'] = processed_df['SESSION'] + processed_df['SUBJECT'].astype(str)


df = dataset_df.merge(processed_df, on='new_col', how='left')

df.drop(['new_col', 'Unnamed: 0'], axis=1, inplace=True)

processed_categories = ['age', 'gender', 'pathological', 
                        'IMPRESSION', 'DESCRIPTION OF THE RECORD',
                        'CLINICAL HISTORY', 'MEDICATIONS', 'INTRODUCTION',
                        'CLINICAL CORRELATION', 'HEART RATE', 'FINDINGS', 'REASON FOR STUDY',
                        'TECHNICAL DIFFICULTIES', 'EVENTS', 'CONDITION OF THE RECORDING',
                        'PAST MEDICAL HISTORY', 'TYPE OF STUDY', 'ACTIVATION PROCEDURES',
                        'NOTE']

# new column REPORT containing all the non-empty processed categories as : CATEGORY: TEXT
df['REPORT'] = ''
for category in processed_categories:
  for i, value in enumerate(df[category]):
    if not pd.isnull(value):
      df.loc[i,'REPORT'] += category + ': ' + str(value) + ', '