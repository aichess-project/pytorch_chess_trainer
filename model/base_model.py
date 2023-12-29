import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def init_net(self, net_config, converter):
        raise Exception("Not implemented")

    def init_weights(self):
        raise Exception("Not implemented")

    def align_eval(self, eval):
        raise Exception("Not implemented")
    
    def get_name(self):
        raise Exception("Not implemented")
    
    def get_hash_value(self):
        raise Exception("Not implemented")
    
    def forward(self, x_tensor):
        raise Exception("Not implemented")
    
    def load(self, filename):
        raise Exception("Not implemented")

    def save(self, filename):
        raise Exception("Not implemented")

