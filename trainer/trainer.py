import logging
from data_classes.dc_dataloader import DC_DataLoad

import torch
import torch.nn as nn
import torch.optim as optim

class Chess_Trainer():

  def __init__(self, trainer_config, data_loaders, machine):       
    self.data_loaders = data_loaders
    self.trainer_config = trainer_config
    self.machine = machine
    self.init_components()

  def init_components(self):
    optimizer_class = getattr(optim, self.trainer_config.optimizer_name)
    criterion_class = getattr(nn, self.trainer_config.criterion_name)

    self.optimizer = optimizer_class(self.machine.parameters(), lr=self.trainer_config.learning_rate, weight_decay=self.trainer_config.weight_decay)
    self.criterion = criterion_class()
    self.shuffle = self.trainer_config.shuffle
    torch.backends.cudnn.enabled = self.trainer_config.cudnn_enabled
    self.threshold = self.trainer_config.test_threshold

  def run_machine(self, type):
    if type == DC_DataLoad.training():
      self.machine.train()
    else:
       self.machine.eval()
    running_loss = 0.0
    for x, y in self.data_loaders[type]:
      outputs = self.machine(x)
      if type == DC_DataLoad.testing():
        absolute_difference = torch.abs(y - outputs)
        loss = torch.sum(absolute_difference > self.threshold)
      else:
        loss = self.criterion(outputs, y)
      if type == DC_DataLoad.training():
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      running_loss += loss.item()
      self.first = False
    return running_loss
  
  def do_training(self, num_epochs = 10):
    #
    # Load Net File
    #
    for epoch in range(num_epochs):
        running_loss = self.run_machine(type = DC_DataLoad.training())
        logging.info(f'Training Results! Epoch:{epoch}, Running Loss:{int(running_loss)}, Items: {int(len(self.data_loaders[DC_DataLoad.training()]))}, Avg: {round(running_loss/(len(self.data_loaders[DC_DataLoad.training()])),2)}')
        with torch.no_grad():
          running_loss = self.run_machine(type = DC_DataLoad.validating())
          logging.info(f'Validation Results! Epoch:{epoch}, Running Loss:{int(running_loss)}, Items: {int(len(self.data_loaders[DC_DataLoad.training()]))}, Avg: {round(running_loss/(len(self.data_loaders[DC_DataLoad.training()])),2)}')
          running_loss = self.run_machine(type = DC_DataLoad.testing())
          logging.info(f'Testing Results! Epoch:{epoch}, Running Loss:{int(running_loss)}, Items: {int(len(self.data_loaders[DC_DataLoad.training()]))}, Avg: {round(running_loss/(len(self.data_loaders[DC_DataLoad.training()])),2)}')
    #
    # Save Net File
    #  