import os
import scipy.io as sio
import pandas as pd

class PatientTableData:
    def __init__(self, table_data_root):
        self.all_mats = []
        self.all_mats_path = []
        self.table_data_root = table_data_root
        for file in os.listdir(table_data_root):
            if not file.endswith('.mat'):
                continue
            self.all_mats.append(self.read_mat(table_data_root + file))
            self.all_mats_path.append(file)

    def read_mat(self, patient_table_path):
        """
        return a dataframe of the patient table
        """
        data = sio.loadmat(patient_table_path)[patient_table_path.split('/')[-1].split('.')[0]] # (1, 106)
        columns = list(data.dtype.names)

        data_list = [None] * len(data[0, :])
        for idx, row in enumerate(data[0, :]):
            row_data = [x[0] if len(x) > 0 else '' for x in row]
            data_list[idx] = row_data 
        df = pd.DataFrame(data_list, columns=columns)
        return df


    def if_name_in_table(self, name, path):
        """
        check if the name is in the table file
        """
        ind = self.all_mats_path.index(path)
        df = self.all_mats[ind]
        table_name = df['name'].unique()
        return name in table_name
    

    def get_feature_person(self, name, feature, path):
        ind = self.all_mats_path.index(path)
        df = self.all_mats[ind]
        return df[df['name'] == name][feature].values[0] if name in df['name'].values else None