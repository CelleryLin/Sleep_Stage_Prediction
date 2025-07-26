# use absolute path

### PREPROCESSING CONFIGURATION
data_root = 'G:/Cellery/merry/data/'
table_data_root = 'G:/Cellery/merry/data/SM_patient_data/'
patient_lut_file = 'G:/Cellery/merry_submit/data/patient_lut.csv'
combined_data_folder = './data/combined/'
clf_dataset_root = 'F:/Cellery/merry/data/label_window/'

### CLASSIFICATION TRAINING CONFIGURATION

local_clf_ds_path = 'F:/Cellery/merry/data/label_window/ECG_Rate_000100_pos/'

# Model output paths
local_output_root = 'G:/Cellery/merry_submit/Sleep_stage_classifier/Local_classification/output/'
global_output_root = 'G:/Cellery/merry_submit/Sleep_stage_classifier/Global_classification/output/'

# Local classification model paths
model_path_dict = {
    '010000': local_output_root + '20250723130353_010000/',
    '111000': local_output_root + '20250723143930_111000/', 
    '1n0000': local_output_root + '20250723162924_1n0nnn/',
    # 'nnn100': local_output_root + '20250723171908_nnn100/',
    '000100': local_output_root + '20250726112253_000100/',
}