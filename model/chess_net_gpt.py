from libs.chess_lib import Chess_Lib
from config.net_config import load_net_config_from_yaml, save_net_config_to_yaml
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import logging
import hashlib
from model.simple_krk import ChessBaseNet

class ChessNetGPT(ChessBaseNet):

    def init_net(self, net_config):
        logging.info(f"Net Config: {net_config}")
        self.ft = nn.Linear(net_config.nodes["NUM_FEATURES"], net_config.nodes["M"])
        self.l1 = nn.Linear(net_config.nodes["M"], net_config.nodes["K"])
        self.l2 = nn.Linear(net_config.nodes["K"], 1)
        # Initialize the weights
        self.init_weights()
    
    def init_weights(self):
        for layer in [self.ft, self.l1, self.l2]:
            if hasattr(layer, 'weight'):
                init.xavier_uniform_(layer.weight)

    def get_name(self):
        return "Chess-Net ChatGPT Version 0.1"
    
    def forward(self, x):
        x = F.relu(self.ft(x))
        x = F.relu(self.l1(x))
        x = torch.tanh(self.l2(x)) * 100
        return x
    
