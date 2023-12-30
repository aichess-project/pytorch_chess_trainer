import torch
import logging
from model.chess_base_net import ChessBaseNet

class SimpleKRkNet(ChessBaseNet):
    
    def forward(self, x):
        ft_x = self.ft(x)
        # Run the linear layers and use clamp_ as ClippedReLU
        l1_x = torch.clamp(ft_x, 0.0, 1.0)
        l2_x = torch.clamp(self.l1(l1_x), 0.0, 1.0)
        # ChatGPT
        l3_x = torch.clamp(self.l2(l2_x), 0.0, 1.0)  # Output from the new layer
        eval = torch.clamp(self.l3(l3_x), 0.0, 1.0)
        return self.align_eval(eval)
    