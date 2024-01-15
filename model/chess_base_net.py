from libs.chess_lib import Chess_Lib
from config.net_config import load_net_config_from_yaml, save_net_config_to_yaml
import torch
import torch.nn as nn
import torch.nn.init as init
import logging
import hashlib
from model.base_model import BaseModel

class ChessBaseNet(BaseModel):

    def __init__(self):
        super(ChessBaseNet, self).__init__()
        self.first = True

    def init_net(self, net_config):
        logging.info(f"Net Config: {net_config}")
        self.ft = nn.Linear(net_config.nodes["NUM_FEATURES"], net_config.nodes["M"])
        self.l1 = nn.Linear(net_config.nodes["M"], net_config.nodes["N"])
        self.l2 = nn.Linear(net_config.nodes["N"], net_config.nodes["K"])
        self.l3 = nn.Linear(net_config.nodes["K"], 1)
        # Initialize the weights
        self.init_weights()

    def init_weights(self):
        for layer in [self.ft, self.l1, self.l2, self.l3]:
            if hasattr(layer, 'weight'):
                init.xavier_uniform_(layer.weight)

    def align_eval(self, eval):
      eval = (eval - 0.5) * 2 * Chess_Lib.MAX_MATE
      return eval.floor()
    
    def get_name(self):
        return "Chess-Basenet_Version_0.1"
    
    def get_hash_value(self):
        # Serialize the model's state_dict into a string
        state_dict_str = str(self.state_dict()) + str(self.named_parameters)
        # Compute the SHA-256 hash of the state_dict string
        sha256_hash = hashlib.sha256(state_dict_str.encode()).hexdigest()
        # Convert the hexadecimal hash to an integer
        hash_value = abs(int(sha256_hash, 16))
        return hash_value
    
    def forward(self, x):
        raise Exception("Not implemented")
    