import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from Inference_pipeline.src.pipeline import Pipeline
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from Sleep_stage_classifier.Local_classification.Models.model import ConvTran
from Sleep_stage_classifier.Global_classification.Model.resnet50 import ResNet50, UNet
from stage_code_cvt import stage_code_global
import file_paths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
# Import the function from centralized utilities
from _utils.model_utils import load_data
 
# load local model
# Import the function from centralized utilities
from _utils.model_utils import load_local_model

# load global model
# Import the function from centralized utilities
from _utils.model_utils import load_global_model

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