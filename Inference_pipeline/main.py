import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pipeline import Pipeline
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from Sleep_stage_classifier.Local_classification.Models.model import ConvTran
from Sleep_stage_classifier.Global_classification.Model.resnet50 import ResNet50, UNet
from stage_code_cvt import stage_code_global
import file_paths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
def load_data(data_base_dir, filename_list):
    stage_code = stage_code_global
    all_time_series = []
    all_stages = []
    for file in tqdm(filename_list):
        data = np.load(data_base_dir + file, allow_pickle=True)
        data_dict = data['data'].item()

        if data_dict['BPM'] is None or data_dict['STAGE'] is None:
            continue

        time_series = np.array(data_dict['BPM'])[::128] # transform to 1 beat per sec
        stage = np.array(data_dict['STAGE'][:, 0])[::128]

        # encode stage
        stage = np.vectorize(stage_code.get)(stage.astype(str))

        all_time_series.append(time_series)
        all_stages.append(stage)

    return all_time_series, all_stages
 
# load local model
def load_local_model(model_path_dict):
    """
    4 models need to be loaded:
    1. REM - NREM (010000)
    2. Wake, REM, N1 - N2, N3 (111000)
    3. Wake - N1 (1n0nnn)
    4. N2 - N3 (nnn100)
    model_path_dict: dict, must contain ['010000', '111000', '1n0nnn', 'nnn100']
    """

    models = {}
    training_infos = {}
    test_resultses = {}
    for k, v in model_path_dict.items():

        if not os.path.exists(v):
            raise ValueError(f"Model path {v} does not exist")
        
        training_info_path = os.path.join(v, 'training_info.npz')
        test_results_path = os.path.join(v, 'test_results.npz')
        training_info = np.load(training_info_path, allow_pickle=True)
        test_results = np.load(test_results_path, allow_pickle=True)
        config = training_info['config'].item()

        models[k] = ConvTran(config, num_classes=1).to(device)
        models[k].load_state_dict(torch.load(v+'model.pth'))
        models[k].to(device)
        training_infos[k] = training_info
        test_resultses[k] = test_results

    return models, training_infos, test_resultses

# load global model
def load_global_model(model_dir):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the saved model
        model_type: Type of model ('resnet' or 'unet')
        n_channels: Number of input channels
        n_classes: Number of output classes
        device: Device to load the model on
    
    Returns:
        Loaded PyTorch model
    """
    training_info_path = os.path.join(model_dir, 'training_info.npz')
    
    if not os.path.exists(training_info_path):
        raise FileNotFoundError(f"Training info file not found: {training_info_path}")
    
    print(f"Loading training configuration from: {training_info_path}")
    training_info = np.load(training_info_path, allow_pickle=True)

    model_type = training_info['model_type'].item()
    n_output_classes = training_info['n_output_classes'].item()
    

    # Determine number of input channels from model path dict
    n_input_channels = len(file_paths.model_path_dict)

    # Initialize model
    if model_type.lower() == 'resnet':
        model = ResNet50(in_channels=n_input_channels, classes=300).to(device)
    elif model_type.lower() == 'unet':
        model = UNet(n_channels=n_input_channels, n_classes=n_output_classes, bilinear=True).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load weights
    model.load_state_dict(torch.load(model_dir+'model.pth', map_location=device))
    print(f"Model loaded from: {model_dir+'model.pth'}")

    return model

if __name__ == '__main__':
    # prepare data
    data_base_dir = file_paths.combined_data_folder
    patient_lut = pd.read_csv(file_paths.patient_lut_file)
    train_data_filenames = patient_lut[patient_lut['split'] == 'train']['name'].apply(lambda x: x+'_combined.npz').tolist()
    test_data_filenames = patient_lut[patient_lut['split'] == 'test']['name'].apply(lambda x: x+'_combined.npz').tolist()


    local_models, _, test_resultses = load_local_model(file_paths.model_path_dict)
    global_model = load_global_model(file_paths.global_output_root + '20250724172221/')
    all_time_series, all_stages = load_data(data_base_dir, test_data_filenames)
    
    # create pipeline
    pipeline = Pipeline(local_models, global_model, test_resultses, all_time_series, all_stages)
    pipeline()
    
    # plot pipeline.global_pred
    # for i in range(len(pipeline.global_pred)):
    #     time_series = pipeline.all_time_series[i]
    #     mat = pipeline.global_pred[i]
    #     result = np.full_like(mat, -1)
    #     for col in range(mat.shape[1]):
    #         col_values = mat[:, col]
    #         non_neg_values = col_values[col_values != -1]
    #         result[:len(non_neg_values), col] = non_neg_values

    #     gt = pipeline.gt[i]
    #     maj_pred = pipeline.majority_pred[i] * 0.8

    #     x = np.linspace(0, len(result), len(time_series))
    #     fig, ax = plt.subplots(3, 1)
    #     fig.set_size_inches(10, 5)
    #     ax[0].plot(x, time_series)
    #     ax[1].imshow(result[:200,:], aspect='auto')
    #     ax[2].plot(gt, label='gt')
    #     ax[2].plot(maj_pred, label='maj_pred')
    #     ax[0].set_ylabel('Time Series')
    #     ax[1].set_ylabel('Global Prediction')
    #     ax[2].set_ylabel('Majority Vote')
    #     ax[2].legend()
        
    #     # save as jpg
    #     plt.savefig(f'G:/Cellery/merry/Combine_local_and_global/image/out_{i}.jpg')


    # evaulate
    from sklearn.metrics import matthews_corrcoef, confusion_matrix
    gt = np.concatenate(pipeline.gt)
    pred = np.concatenate(pipeline.majority_pred)
    print(matthews_corrcoef(gt, pred))
    print(confusion_matrix(gt, pred))