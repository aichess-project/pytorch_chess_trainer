import torch
import torch.nn as nn
import os
import logging

class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def init_net(self, net_config):
        raise Exception("Not implemented")

    def init_weights(self):
        raise Exception("Not implemented")

    def align_eval(self, eval):
        raise Exception("Not implemented")
    
    def get_name(self):
        return 'model'
    
    def get_hash_value(self):
        raise Exception("Not implemented")
    
    def forward(self, x_tensor):
        raise Exception("Not implemented")
    
    def save_model(self, filepath=None):
        if filepath is None:
            filepath = self.get_name() + '.pth'
        # Save the state dictionary to a file
        torch.save(self.state_dict(), filepath)
        logging.info(f'Model saved to {filepath}')

    def load_model(self, filepath=None):
        if filepath is None:
            filepath = self.get_name() + '.pth'
        # Load the state dictionary from a file
        self.load_state_dict(torch.load(filepath))
        logging.info(f'Model loaded from {filepath}')

