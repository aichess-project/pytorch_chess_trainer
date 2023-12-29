from halfkp_libs.chess_lib import Chess_Lib
import torch
import torch.nn as nn
import torch.nn.init as init
import logging
import hashlib
from model.chess_base_net import ChessBaseNet

class SimpleKRkNet(ChessBaseNet):
    def __init__(self, NUM_FEATURES = 3, M = 256, N = 16, K = 4):
        #
        # Features: K, R, k, turn
        #
        super(SimpleKRkNet, self).__init__()
        
    # The inputs are a whole batch!
    # `stm` indicates the whether white is the side to move. 1 = true, 0 = false.
    def forward(self, x):
        white, black, turn = self.converter.get_input_tensor(x)
        input = torch.tensor([white, black, turn]).to(torch.float32)
        ft_x = self.ft(input)
        # Run the linear layers and use clamp_ as ClippedReLU
        l1_x = torch.clamp(ft_x, 0.0, 1.0)
        l2_x = torch.clamp(self.l1(l1_x), 0.0, 1.0)
        # ChatGPT
        l3_x = torch.clamp(self.l2(l2_x), 0.0, 1.0)  # Output from the new layer
        eval = torch.clamp(self.l3(l3_x), 0.0, 1.0)
        return self.align_eval(eval)
    