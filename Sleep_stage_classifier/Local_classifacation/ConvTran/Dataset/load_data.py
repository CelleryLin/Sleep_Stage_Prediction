import os
import numpy as np
from tqdm import tqdm

def load_data(
    data_path, 
    patient_lut,
    data_cache_folder, 
    drop_unacceptable=True,
    use_cache=True
):
    cache_file_name = 'data.npz'
    if use_cache and os.path.exists(data_cache_folder + cache_file_name):
        print("Loading data from cache")
        data = np.load(data_cache_folder + cache_file_name, allow_pickle=True)
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        name_train = data['name_train']
        name_test = data['name_test']
    else:
        print("Loading data from source and saving cache")
        X_train, y_train, X_test, y_test, name_train, name_test = \
            _load_data(data_path, patient_lut)
        os.makedirs(data_cache_folder, exist_ok=True)
        np.savez(data_cache_folder + cache_file_name, 
            X_train=X_train, 
            y_train=y_train, 
            X_test=X_test, 
            y_test=y_test, 
            name_train=name_train, 
            name_test=name_test
        )

    # Filter out invalid labels
    X_train = X_train[y_train != -1]
    y_train = y_train[y_train != -1]
    X_test = X_test[y_test != -1]
    y_test = y_test[y_test != -1]

    return X_train, y_train, X_test, y_test, name_train, name_test

def _load_data(data_path, patient_lut):
    """Load training and test data from TSV files."""
    X_train, y_train, X_test, y_test = [], [], [], []
    name_train, name_test = [], []
    
    for i in tqdm(range(len(patient_lut)), desc="Loading data"):
        name = patient_lut.iloc[i]['name']
        split = patient_lut.iloc[i]['split']

        if not os.path.exists(os.path.join(data_path, name+'_combined.tsv')):
            continue
        
        sample = np.loadtxt(os.path.join(data_path, name+'_combined.tsv'), delimiter='\t')
        y = sample[:, 0]
        X = sample[:, 1:]
        name_list = [name] * y.shape[0]

        if split == 'train':
            X_train.extend(X)
            y_train.extend(y)
            name_train.extend(name_list)
        else:
            X_test.extend(X)
            y_test.extend(y)
            name_test.extend(name_list)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    name_train = np.array(name_train)
    name_test = np.array(name_test)

    return X_train, y_train, X_test, y_test, name_train, name_test