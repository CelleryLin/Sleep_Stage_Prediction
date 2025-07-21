"""
make dataset for classification task

input:
    raw npy (stage and ecg)
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
            'BPM': bpm,
            'STAGE': stage,
        },
    }
output:
    a window and its label per row, saved as tsv file
    (label feature1 feature2 ...)

methology:
We uses multiple epochs to predict the central epoch's sleep stage.
For example, For m=5, Stage_n is predicted from BPM series of 
[epoch_n-2, epoch_n-1, epoch_n, epoch_n+1, epoch_n+2].

process:
1. load necessary files
2. down sampling to fs=1
3. windowing
4. extract stage from the center epoch
5. label the window with the stage code
6. save as tsv file
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import file_paths
import stage_code_cvt
from stage_code_cvt import stage_code, stage_code_str

##### configurations #####
"""
m: number of windows (must be odd)
record_pos: Whether to include the relative position of each window within the entire signal as a feature.
"""

record_pos = True
m = 5 # must be odd

##########################

data_root = file_paths.combined_data_folder
output_root = file_paths.clf_dataset_root + f'ECG_Rate_{stage_code_str}_{"pos" if record_pos else ""}/'

# check data_root exists and has npzs
if not os.path.exists(data_root):
    raise FileNotFoundError(f'Data root not found: {data_root}, have you run combine_stage_ecg.py yet?')
else:
    npz_files = [f for f in os.listdir(data_root) if f.endswith('.npz')]
    if not npz_files:
        raise FileNotFoundError(f'No .npz files found in {data_root}, please check the directory.')

if not os.path.exists(output_root):
    os.makedirs(output_root)
else:
    if input(f'Output directory {output_root} already exists, overwriting? (y/n): ').strip().lower() != 'y':
        print('Exiting without overwriting.')
        sys.exit(0)

if __name__ == '__main__':
    stage_code_cvt.info(stage_code)

    fs=1 # do not modify.
    epoch_len = 30 #s, do not modify.

    for file in tqdm(os.listdir(data_root)):
        if not file.endswith('.npz'):
            continue
        data = np.load(data_root + file, allow_pickle=True)
        data_dict = data['data'].item()
        time_series = np.array(data_dict['BPM'])[::128]
        stage = np.array(data_dict['STAGE'][:, 0])[::128]

        # get time series positional
        if record_pos:
            pos = np.linspace(0, 1, len(time_series))

        if np.all(time_series == np.nan):
            continue
        full_epochs_num = time_series.shape[0] // fs // epoch_len
        all_window_data = np.zeros((full_epochs_num - (m-1), m*fs*epoch_len+1))

        # windowing
        # for win in range(full_epochs_num - (m-1)):
        #     ecg_window = time_series[win*fs*epoch_len:(win+m)*fs*epoch_len]
        #     stage_label = stage[(win+m//2)*fs*epoch_len: (win+m//2+1)*fs*epoch_len]
        #     assert np.all(stage_label == stage_label[0])
        #     stage_label = stage_code[str(stage_label[0])]
        #     all_window_data[win,:] = np.concatenate(([stage_label], ecg_window)).reshape(1,-1)

        # Precompute the indices for the windows
        window_indices = np.arange(full_epochs_num - (m - 1)) * fs * epoch_len
        window_slices = np.array([np.arange(i, i + m * fs * epoch_len) for i in window_indices])

        # Extract the ECG windows using advanced indexing
        ecg_windows = time_series[window_slices].reshape(full_epochs_num - (m - 1), -1)

        # Extract the stage labels for the center epochs
        center_epoch_indices = window_indices + (m // 2) * fs * epoch_len
        stage_labels = stage[center_epoch_indices]
        if record_pos:
            pos_labels = pos[center_epoch_indices]

        # Ensure all stage labels in the center epoch are the same
        # [1, 2, 3] == [[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3]]
        all_middle_stage = stage[window_slices[:, m // 2 * fs * epoch_len: (m // 2 + 1) * fs * epoch_len]]
        assert np.all([np.all(all_middle_stage[i,:] == stage_labels[i]) for i in range(all_middle_stage.shape[0])])

        # Map stage labels to their corresponding codes
        stage_labels = np.vectorize(stage_code.get)(stage_labels.astype(str))


        # Combine stage labels and ECG windows
        # stage_labels (n_windows,)
        # ecg_windows (n_windows, m * fs * epoch_len)
        if record_pos:
            all_window_data = np.hstack((stage_labels[:, None], pos_labels[:, None], ecg_windows))
        else:
            all_window_data = np.hstack((stage_labels[:, None], ecg_windows))

        np.savetxt(output_root + file.split('.')[0] + '.tsv', all_window_data, fmt='%s', delimiter='\t')
    
    print(f'Dataset saved to {output_root}.')