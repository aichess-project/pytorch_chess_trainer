from halfkp_libs.chess_lib import Chess_Lib
import torch
import torch.nn as nn
import torch.nn.init as init
import logging
import hashlib
from net.chess_net_base import ChessBaseNet

class SimpleKRkNet(ChessBaseNet):
    def __init__(self, NUM_FEATURES = 3, M = 256, N = 16, K = 4):
        #
        # Features: K, R, k, turn
        #
        super(SimpleKRkNet, self).__init__()
        self.ft = nn.Linear(NUM_FEATURES, M)
        logging.info(f"Self.FT Weights: {self.ft.weight.dtype}")
        self.l1 = nn.Linear(M, N)
        self.l2 = nn.Linear(N, K)
        # ChatGPT
        self.l3 = nn.Linear(K, 1)  # New linear layer to reduce from K to 1

        # Initialize the weights
        self.init_weights()

    def init_weights(self):
        for layer in [self.ft, self.l1, self.l2, self.l3]:
            if hasattr(layer, 'weight'):
                init.xavier_uniform_(layer.weight)

    # The inputs are a whole batch!
    # `stm` indicates the whether white is the side to move. 1 = true, 0 = false.
    def forward(self, white, black, turn):
        input = torch.tensor([white, black, turn]).to(torch.float32)
        if self.first:
            logging.info(f"Input: {input.size()} {input.dtype}")
        ft_x = self.ft(input)
        if self.first:
            logging.info(f"FT_X: {ft_x.size()}")

        # Run the linear layers and use clamp_ as ClippedReLU
        l1_x = torch.clamp(ft_x, 0.0, 1.0)
        l2_x = torch.clamp(self.l1(l1_x), 0.0, 1.0)
        if self.first:
            logging.info(f"L2_X: {l2_x.size()}")
        # ChatGPT
        l3_x = torch.clamp(self.l2(l2_x), 0.0, 1.0)  # Output from the new layer
        if self.first:
            logging.info(f"L3_X: {l3_x.size()}")
        eval = torch.clamp(self.l3(l3_x), 0.0, 1.0)
        if self.first:
            logging.info(f"Eval: {eval.size()}")
        self.first = False
        return self.align_eval(eval)
    