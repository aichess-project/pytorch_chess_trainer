from libs.log_lib import setup_logging
from trainer.trainer import Chess_Trainer
from data_loader.base_loader import Base_Dataset
from config.trainer_config import load_trainer_config_from_yaml
from config.net_config import get_net, load_net_config_from_yaml
from config.dl_config import load_dl_config_from_yaml
from config.conv_config import get_converter, load_conv_config_from_yaml
from trainer.trainer import Chess_Trainer
import logging


def create_net():
    net_config = load_net_config_from_yaml("net_config.yaml")
    logging.info(f"Net Config: {net_config}")
    net_class = get_net(net_config)
    net_class.init_net(net_config)
    return net_class

def create_converter():
    conv_config = load_conv_config_from_yaml("conv_config.yaml")
    logging.info(f"Converter Config: {conv_config}")
    return get_converter(conv_config)

def create_dataloader(training_steps, converter):
    datasets = {}
    dl_config = load_dl_config_from_yaml("dl_config.yaml")
    logging.info(f"DL Config: {dl_config}")
    for step in training_steps:
      dataset = Base_Dataset(dl_config, converter, step)
      datasets[step] = dataset
    return datasets

def create_trainer(trainer_config, data_sets, net):
    trainer = Chess_Trainer(trainer_config, data_sets, net)
    return trainer    

def main():
    trainer_config = load_trainer_config_from_yaml("trainer_config.yaml")
    logging.info(f"Trainer Config: {trainer_config}")
    converter = create_converter()
    net = create_net()
    data_sets = create_dataloader(trainer_config.training_steps, converter)
    trainer = create_trainer(trainer_config, data_sets, net)
    trainer.do_training(num_epochs=trainer_config.epochs)

if __name__ == '__main__':
    setup_logging()
    logging.info("Start Training")
    main()
    logging.info("End Training")
