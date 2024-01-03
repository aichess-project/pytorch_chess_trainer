import torch.nn as nn
import torch
from model.base_model import BaseModel

class HW_Model(BaseModel):

    def __init__(self):
        super(HW_Model, self).__init__()

    def init_net(self, net_config):
        self.fc = nn.Sequential(
            nn.Linear(4,5),
            nn.ReLU(),
            nn.Linear(5,1)
        )

    def init_weights(self):
        pass
    
    def forward(self, x_tensor):
        y = self.fc(x_tensor)
        return torch.squeeze(y)
