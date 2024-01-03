import logging
from data_classes.dc_dataloader import DC_DataLoad

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from libs.loss_adjustment import adjust_loss

class Chess_Trainer():

  def __init__(self, trainer_config, data_sets, machine):   
    self.data_loaders = {}
    self.batch_size = trainer_config.batch_size
    for step in trainer_config.training_steps:
      logging.info(f"Length Dataset {step}: {len(data_sets[step])}")
      if step != "test":
        batch_size = self.batch_size
        shuffle = True
      else:
        batch_size = 1
        shuffle = False
      self.data_loaders[step] = DataLoader(data_sets[step], batch_size=batch_size, shuffle=shuffle, num_workers=1)
      logging.info(f"Length Dataloader {step}: {len(self.data_loaders[step])}")
    self.trainer_config = trainer_config
    self.machine = machine
    self.init_components()

  def init_components(self):
    optimizer_class = getattr(optim, self.trainer_config.optimizer)
    criterion_class = getattr(nn, self.trainer_config.criterion)

    self.optimizer = optimizer_class(self.machine.parameters(), lr=self.trainer_config.learning_rate, weight_decay=self.trainer_config.weight_decay)
    self.criterion = criterion_class()
    self.shuffle = self.trainer_config.shuffle
    torch.backends.cudnn.enabled = self.trainer_config.cudnn_enabled
    self.threshold = self.trainer_config.test_threshold

  def run_machine(self, step):
    logging.info(f"Start Machine: {step}")
    if step == "train":
      self.machine.train()
    else:
       self.machine.eval()
    running_loss = 0.0
    for x, y in self.data_loaders[step]:
      logging.debug(f"RUN Machine y: {y} {y.shape} {y.dtype} X: {x} {x.shape} {x.dtype}")
      outputs = self.machine(x)
      logging.debug(f"RUN Machine out: {outputs} {outputs.shape} {outputs.dtype}")
      if step == "test":
        absolute_difference = torch.abs(y - outputs)
        loss = torch.sum(absolute_difference > self.threshold)
      else:
        loss = self.criterion(outputs, y)
      if step == "train":
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      running_loss += adjust_loss(loss.item(), step, self.trainer_config)
    return running_loss
  
  def log_results(self, run_nr, step, epoch, running_loss, nr_items):
    if step == "train":
      step_str = "Training"
    elif step == "valid":
      step_str = "Validation"
    else:
      step_str = "Testing"
    logging.info(f'{step_str} Results! Run: {run_nr} Epoch:{epoch}, Running Loss:{int(running_loss)}, Items: {nr_items}, Avg: {round(running_loss/(nr_items),2)}')

  def do_training(self, score, num_epochs = 10):
    #
    # Load (best)Net File
    #
    for epoch in range(num_epochs):
        score.next_run += 1
        running_loss = self.run_machine(step = "train")
        nr_items = int(len(self.data_loaders["train"]))
        avg = round(running_loss/(nr_items),2)
        self.log_results(score.next_run, "train", epoch, running_loss, nr_items)
        if score.best_score > avg and avg > 0.0:
          score.best_score = avg
          score.best_run = score.next_run
          logging.info(f"New best Score: {score.best_score} {score.best_run}")
          #
          # Save Net
          #
        with torch.no_grad():
          running_loss = self.run_machine(step = "valid")
          nr_items = int(len(self.data_loaders["valid"]))
          self.log_results(score.next_run, "valid", epoch, running_loss, nr_items)
          running_loss = self.run_machine(step = "test")
          nr_items = int(len(self.data_loaders["test"]))
          self.log_results(score.next_run, "test", epoch, running_loss, nr_items)
    return score