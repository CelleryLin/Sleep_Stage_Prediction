"""
Concat combine stage and ECG data with center alignment.

This script:
1. Copies stage data fs*30=2840 times
2. Concatenates them
3. If stage data is longer than ECG data, zero pads the ECG data
4. If ECG data is longer than stage data, trims the ECG data
5. Gets BPM and signal quality using neurokit2
6. Saves the combined data as NPZ file using multiprocessing

Output data structure:
{
    'original_filename': {
        'ECG': ecg_path,
        'STAGE': stage_path,
        'AHI': ahi_path,
    },
    'scalar_features': {
        'AHI': ahi,
        'quality': quality, # Unacceptable, Barely acceptable, Excellent
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
import scipy.io as sio
from tqdm import tqdm
import neurokit2 as nk
import pandas as pd
import multiprocessing as mp
from functools import partial

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from _utils.PatientTableData import PatientTableData
import file_paths

def get_bpm(ecg: np.ndarray, fs=128, method='neurokit'):
    """
    Calculate heart rate (BPM) from ECG signal.
    
    Args:
        ecg: ECG signal array
        fs: Sampling frequency (Hz)
        method: Method for ECG processing ('neurokit', 'pantompkins1985', 'hamilton2002', 'elgendi2010', 'engzeemod2012')
    
    Returns:
        tuple: (bpm array, signal quality)
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

        # Calculate quality (0~1)
        quality = nk.ecg.ecg_quality(ecg_cleaned, rpeaks=info["ECG_R_Peaks"], sampling_rate=fs, method='zhao2018')

        return bpm, quality
   
    except Exception as e:
        return None, 'Unacceptable'

def combine_stage_ecg(ecg_path, stage_path, fs=128):
    """
    Combine ECG and sleep stage data, aligning them in time.
    
    Args:
        ecg_path: Path to ECG .mat file
        stage_path: Path to sleep stage .mat file
        fs: Sampling frequency (Hz)
        
    Returns:
        tuple: (ecg, bpm, stage, quality)
    """
    # Verify files are for the same patient
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

    bpm, quality = get_bpm(ecg, fs)
    if bpm is not None:
        assert stage.shape[0] == ecg.shape[0] == bpm.shape[0], f'{ecg_path} and {stage_path} not aligned'

    return ecg, bpm, stage, quality

def process_patient(row, data_root, output_root, patient_table_data):
    """
    Process a single patient's data.
    
    Args:
        row: Row from the patient LUT DataFrame
        data_root: Root directory for data
        output_root: Output directory for combined data
        patient_table_data: PatientTableData instance
    """
    name = row['name']
    ecg_path = row['ecg_filepath']
    stage_path = row['stage_filepath']
    table_filepath = row['table_filepath']
    
    if pd.isna(table_filepath):
        ahi = None
    else:
        ahi = patient_table_data.get_feature_person(name, 'reportrdi', table_filepath)
    
    if pd.isna(ecg_path) or pd.isna(stage_path):
        ecg = None
        bpm = None
        stage = None
        quality = None
    else:
        ecg, bpm, stage, quality = combine_stage_ecg(
            data_root + ecg_path,
            data_root + stage_path,
        )
    
    combined = {
        'original_filename': {
            'ECG': ecg_path,
            'STAGE': stage_path,
            'AHI': row['table_filepath'],
        },
        'scalar_features': {
            'AHI': ahi,
            'quality': quality,
        },
        'data': {
            'ECG': ecg,
            'BPM': bpm,
            'STAGE': stage,
        },
    }
    
    output_path = os.path.join(output_root, f'{name}_combined.npz')
    np.savez(output_path, **combined)
    return output_path

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
    
    # Determine number of processes (adjust based on your CPU)
    num_processes = min(mp.cpu_count(), 16)
    print(f"Processing with {num_processes} processes")
    
    # Create a partial function with fixed arguments
    process_func = partial(
        process_patient,
        data_root=data_root,
        output_root=output_root,
        patient_table_data=patient_table_data
    )
    
    # Create a list of rows to process
    rows_to_process = [row for _, row in patient_lut.iterrows()]
    
    # Process patients in parallel using multiprocessing
    with mp.Pool(processes=num_processes) as pool:
        # Map rows to the process function and track with tqdm
        results = list(tqdm(
            pool.imap(process_func, rows_to_process),
            total=len(rows_to_process),
            desc="Processing patients"
        ))