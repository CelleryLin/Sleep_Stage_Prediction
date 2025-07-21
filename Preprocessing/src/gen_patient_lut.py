"""
This script create a patient look up table (LUT) recording all patient names and their corresponding ECG, stage, and table files.
"""

import os
import sys
import scipy.io as sio
import pandas as pd
import re
from tqdm import tqdm
from _utils.PatientTableData import PatientTableData

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import file_paths

matching_list = {
    '2007_CPAP': ('ECG_2007_CPAP', 'STAGE_2007_CPAP', 'SM07Cpap'),
    '2007_NOCPAP': ('ECG_2007_NOCPAP', 'STAGE_2007_NOCPAP', 'SM07'),
    '2008_CPAP': ('ECG_2008_CPAP', 'STAGE_2008_CPAP', 'SM08Cpap'),
    '2008_NOCPAP': ('ECG_2008_NOCPAP', 'STAGE_2008_NOCPAP', 'SM08'),
    '2009_CPAP': ('ECG_2009_CPAP', 'STAGE_2009_CPAP', 'SM09Cpap'),
    '2009_NOCPAP': ('ECG_2009_NOCPAP', 'STAGE_2009_NOCPAP', 'SM09'),
}

if __name__ == '__main__':

    data_root = file_paths.data_root
    table_data_root = file_paths.table_data_root
    output_file = file_paths.patient_lut_file

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    all_patient_name = []
    all_patient_ecg_files = []
    all_patient_stage_files = []
    all_patient_table_files = []

    patient_table_data = PatientTableData(table_data_root)

    for key, value in matching_list.items():
        ecg_files = os.listdir(os.path.join(data_root, value[0]))
        stage_files = os.listdir(os.path.join(data_root, value[1]))
        table_files = table_data_root + value[2] + '.mat'


        ecg_files_name = list(map(lambda x: x.split('_')[0], ecg_files))
        stage_files_name = list(map(lambda x: x.split('_')[0], stage_files))
        df = patient_table_data.read_mat(table_files)
        table_name = df['name'].unique()

        # all patient name is the union of 3 sets
        
        all_patient_name.extend(ecg_files_name)
        all_patient_name.extend(stage_files_name)
        all_patient_name.extend(table_name)

        all_patient_ecg_files.extend([value[0] + '/' + f for f in ecg_files])
        all_patient_stage_files.extend([value[1] + '/' + f for f in stage_files])
        all_patient_table_files.append(value[2] + '.mat')

    all_patient_name = list(set(all_patient_name))

    # create a DataFrame to store the patient name with their resources.
    # columns: name, ecg_filepath, stage_filepath, table_filepath
    # init with nan
    patient_data = pd.DataFrame(data=None, columns=['name', 'ecg_filepath', 'stage_filepath', 'table_filepath'])
    for i, name in tqdm(enumerate(all_patient_name), total=len(all_patient_name)):
        ecg_filepath = list(filter(lambda v: name in v, all_patient_ecg_files))
        ecg_filepath = ecg_filepath[-1] if len(ecg_filepath) > 0 else None

        stage_filepath = list(filter(lambda v: name in v, all_patient_stage_files))
        stage_filepath = stage_filepath[-1] if len(stage_filepath) > 0 else None

        table_filepath = list(filter(lambda v: patient_table_data.if_name_in_table(name, v), all_patient_table_files))
        table_filepath = table_filepath[-1] if len(table_filepath) > 0 else None
        
        row = pd.DataFrame([[name, ecg_filepath, stage_filepath, table_filepath]], columns=patient_data.columns)
        patient_data = pd.concat([patient_data, row], axis=0, ignore_index=True)

    # add train/test split column (randomly assign 80% to train and 20% to test)
    patient_data['split'] = 'train'
    train_ind = patient_data.sample(frac=0.8, random_state=42).index
    patient_data.loc[~patient_data.index.isin(train_ind), 'split'] = 'test'

    patient_data.to_csv(output_file, index=False)