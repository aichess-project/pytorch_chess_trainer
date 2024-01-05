import logging
import torch
from torch.utils.data import Dataset
import os, sys
from libs.file_lib import os_path

class Base_Dataset(Dataset):

    def __init__(self):
        pass

    def init(self, dl_config, converter, step):
        self.data_path = os.path.join(os_path(dl_config.data_directory), step, dl_config.filename)
        logging.info(f"Init File: {self.data_path}")
        self.y = []
        self.x = []
        self.load_data(self.data_path, converter, dl_config.delimiter)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def load_data(self, data_path, converter, delimiter):
        try:
            with open(data_path, 'r') as file:
                lines = file.readlines()
                logging.info(f"Length Lines: {len(lines)}")
                for line in lines:
                    # Split the line using the specified delimiter
                    items = line.strip().split(delimiter)
                    x = []
                    for index, item in enumerate(items):
                        if index == len(items)-1:
                            y = float(item)
                        else:
                            x.append(item)
                    self.y.append(converter.get_output_tensor(y))
                    self.x.append(converter.get_input_tensor(x))
        except Exception as e:
            logging.info(f"Exception: {e}")
            sys.exit()