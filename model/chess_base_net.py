from pytorch_chess_trainer.libs.chess_lib import Chess_Lib
import torch
import torch.nn as nn
import torch.nn.init as init
import logging
import hashlib

class ChessBaseNet(nn.Module):
    def __init__(self, NUM_FEATURES = 40960, M = 256, N = 32, K = 32):
        super(ChessBaseNet, self).__init__()
        self.first = True
        self.ft = nn.Linear(NUM_FEATURES, M)
        self.l1 = nn.Linear(2 * M, N)
        self.l2 = nn.Linear(N, K)
        self.l3 = nn.Linear(K, 1)

    def align_eval(self, eval):
      eval = (eval - 0.5) * 2 * Chess_Lib.MAX_MATE
      return eval.floor()
    
    def get_name(self):
        return "Chess-Basenet Version 0.1"
    
    def get_hash_value(self):
        # Serialize the model's state_dict into a string
        state_dict_str = str(self.state_dict()) + str(self.named_parameters)
        # Compute the SHA-256 hash of the state_dict string
        sha256_hash = hashlib.sha256(state_dict_str.encode()).hexdigest()
        # Convert the hexadecimal hash to an integer
        hash_value = abs(int(sha256_hash, 16))
        return hash_value
    
    def forward(self, x_tensor):
        raise Exception("Not implemented")
    
    def load(self, filename):
        raise Exception("Not implemented")

    def save(self, filename):
        raise Exception("Not implemented")

