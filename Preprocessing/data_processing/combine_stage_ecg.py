"""
Concat combine stage and ecg data, **align center**.
1. copy stage data fs*30=2840 times
2. concat them
3. if stage data is longer than ecg data, zero pad the ecg data
4. if ecg data is longer than stage data, trim the ecg data
5. get bpm, filled using cubic spline interpolation.
5. save the combined data as npz file:
{
    'original_filename': {
        'ECG': ecg_path,
        'STAGE': stage_path,
        'AHI': ahi_path,
    },
    'scalar_features': {
        'AHI': ahi,
    },
    'data': {
        'ECG': ecg,
        'BPM': bpm, # same fs as ecg
        'STAGE': stage,
    },
}
"""

import numpy as np
import os
import sys
from biosppy import signals
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.interpolate
import neurokit2 as nk
import pandas as pd
from _utils.PatientTableData import PatientTableData
from _utils import file_paths

def get_bpm(ecg: np.ndarray, fs=128, method='neurokit'):
    """
    method: The processing method used for signal cleaning (using ecg_clean()) and peak detection (using ecg_peaks()). Defaults to 'neurokit'. Available methods are 'neurokit', 'pantompkins1985', 'hamilton2002', 'elgendi2010', 'engzeemod2012'. We aim at improving this aspect to make the available methods more transparent, and be able to generate specific reports. Please get in touch if you are interested in helping out with this.
    """
    try:
        # copy from nk.ecg.ecg_process
        # Sanitize and clean input
        ecg_signal = nk.signal.signal_sanitize(ecg[:,0])
        ecg_cleaned = nk.ecg.ecg_clean(ecg_signal, sampling_rate=fs, method=method)

        # Detect R-peaks
        _, info = nk.ecg.ecg_peaks(
            ecg_cleaned=ecg_cleaned,
            sampling_rate=fs,
            method=method,
            correct_artifacts=True,
        )

        # Calculate heart rate
        bpm = nk.signal.signal_rate(
            info, sampling_rate=fs, desired_length=len(ecg_cleaned)
        )

        return bpm
   
    except Exception as e:
        return np.full_like(ecg, np.nan)

def combine_stage_ecg(ecg_path, stage_path, fs=128):
    """
    ecg_path: str, path to the ecg mat file
    stage_path: str, path to the stage mat file
    """
    assert ecg_path.split('/')[-1].split('_')[0] == \
        stage_path.split('/')[-1].split('_')[0]

    ecg = sio.loadmat(ecg_path)['ECG']
    stage = sio.loadmat(stage_path)['STAGE']
    stage = np.repeat(stage, fs*30, axis=0)
    ecg_len = ecg.shape[0]
    stage_len = stage.shape[0]

    if stage_len > ecg_len: # zero pad ecg data
        diff = stage_len - ecg_len
        pads_front = diff // 2
        pads_back = diff - pads_front
        ecg = np.concatenate((np.zeros((pads_front, 1)), ecg, np.zeros((pads_back, 1))), axis=0)
    elif stage_len < ecg_len: # trim ecg data
        diff = ecg_len - stage_len
        ecg = ecg[diff//2:-(diff-diff//2)]

    bpm = get_bpm(ecg, fs)
    assert stage.shape[0] == ecg.shape[0] == bpm.shape[0], f'{ecg_path} and {stage_path} not aligned'

    return ecg, bpm, stage

if __name__ == '__main__':

    data_root = file_paths.data_root
    output_root = file_paths.combined_data_folder
    table_data_root = file_paths.table_data_root
    patient_table_data = PatientTableData(table_data_root)

    # check lut if exists
    if not os.path.exists(file_paths.patient_lut_file):
        raise FileNotFoundError(f'Patient LUT file not found: {file_paths.patient_lut_file}, please run gen_patient_lut.py first.')
    
    patient_lut = pd.read_csv(file_paths.patient_lut_file)
    

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    for index, row in tqdm(patient_lut.iterrows(), total=len(patient_lut)):
        if row.isnull().values.any():
            continue

        name = row['name']
        ecg_path = row['ecg_filepath']
        stage_path = row['stage_filepath']
        ahi = patient_table_data.get_feature_person(name, 'reportrdi', row['table_filepath'])

        ecg, bpm, stage = combine_stage_ecg(
            data_root + '/' + ecg_path, 
            data_root + '/' + stage_path, 
        )

        combined = {
            'original_filename': {
                'ECG': ecg_path,
                'STAGE': stage_path,
                'AHI': row['table_filepath'],
            },
            'scalar_features': {
                'AHI': ahi,
            },
            'data': {
                'ECG': ecg,
                'BPM': bpm,
                'STAGE': stage,
            },
        }
        np.savez(os.path.join(output_root, f'{name}_combined.npz'), **combined)