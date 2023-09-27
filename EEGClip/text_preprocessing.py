import os
import pandas as pd

def text_preprocessing(description_df:pd.DataFrame, processed_categories:str="all" ):

  if processed_categories == "all": 
    processed_categories = ['age', 'gender', 'pathological', 
                          'IMPRESSION', 'DESCRIPTION OF THE RECORD',
                          'CLINICAL HISTORY', 'MEDICATIONS', 'INTRODUCTION',
                          'CLINICAL CORRELATION', 'HEART RATE', 'FINDINGS', 'REASON FOR STUDY',
                          'TECHNICAL DIFFICULTIES', 'EVENTS', 'CONDITION OF THE RECORDING',
                          'PAST MEDICAL HISTORY', 'TYPE OF STUDY', 'ACTIVATION PROCEDURES',
                          'NOTE']
  elif processed_categories == "none":
    processed_categories = []
    
  else:
    processed_categories = [processed_categories]
  # reports were already painstakingly processed by Gemeinl
  # see https://github.com/gemeinl/auto-eeg-diagnosis-comparison/blob/master/code/physician_reports.ipynb

  current_dir = '/home/jovyan/EEGClip/EEGClip/'
  # dataframe created by the TUH class
  #description_df=pd.read_csv(current_dir + '/tuh_description.csv')
  # processed dataframe
  processed_df = pd.read_csv(current_dir + 'tuh_description_processed.csv')

  # add common column to both dataframes
  description_df['new_col'] = 's' + description_df['session'].astype(str).str.pad(3, fillchar='0')  + '_' + description_df['year'].astype(str) + '_' + description_df['month'].astype(str).str.pad(2, fillchar='0') + '_' + description_df['day'].astype(str).str.pad(2, fillchar='0') + description_df["subject"].astype(str)
  processed_df['new_col'] = processed_df['SESSION'] + processed_df['SUBJECT'].astype(str)


  df = description_df.merge(processed_df, on='new_col', how='left')

  df.drop(['new_col',], axis=1, inplace=True)

  # nreport column aggregating the processed data
  df['report'] = ''
  """
  Trick for excluding categories but still measure accuracy
  for category in ["pathological", "age", "gender", "MEDICATIONS"]:
    for i, value in enumerate(df[category]):
      if not pd.isnull(value):
        df.loc[i,'report'] += category + ': ' + str(value) + ', '
  df['report'] = df['report'] + 'pijule'
 """

  for category in processed_categories:
    for i, value in enumerate(df[category]):
      if not pd.isnull(value):
        df.loc[i,'report'] += category + ': ' + str(value) + ', '
  df['report'].replace('', 'no data', inplace=True)
  return df