from libs.log_lib import setup_logging
from trainer.trainer import Chess_Trainer
from data_loader.base_loader import Base_Dataset
#from config.trainer_config import load_trainer_config_from_yaml
from config.trainer_configlist import load_trainer_config_list_from_yaml, config_iterator
from config.net_config import get_net, load_net_config_from_yaml
from config.dl_config import load_dl_config_from_yaml
from config.conv_config import get_converter, load_conv_config_from_yaml
from trainer.trainer import Chess_Trainer
import logging


def create_net(net_config_file):
    net_config = load_net_config_from_yaml(net_config_file)
    logging.info(f"Net Config: {net_config}")
    net_class = get_net(net_config)
    net_class.init_net(net_config)
    return net_class

def create_converter(conv_config_file):
    conv_config = load_conv_config_from_yaml(conv_config_file)
    logging.info(f"Converter Config: {conv_config}")
    return get_converter(conv_config)

def create_dataloader(dl_config_file, training_steps, converter):
    datasets = {}
    dl_config = load_dl_config_from_yaml(dl_config_file)
    logging.info(f"DL Config: {dl_config}")
    for step in training_steps:
      dataset = Base_Dataset(dl_config, converter, step)
      datasets[step] = dataset
    return datasets

def create_trainer(trainer_config, data_sets, net):
    trainer = Chess_Trainer(trainer_config, data_sets, net)
    return trainer    

def main():
    trainer_config_list = load_trainer_config_list_from_yaml("trainer_config_list.yaml")
    logging.info(f"Trainer Config List: {trainer_config_list}")
    converter = create_converter(trainer_config_list.conv_config)
    net = create_net(trainer_config_list.net_config)
    data_sets = create_dataloader(trainer_config_list.dl_config, trainer_config_list.training_steps, converter)
    for trainer_config in config_iterator(trainer_config_list):
        logging.info(f"Trainer Config: {trainer_config}")    
        trainer = create_trainer(trainer_config, data_sets, net)
        trainer.do_training(num_epochs=trainer_config.epochs)

if __name__ == '__main__':
    setup_logging()
    logging.info("Start Training")
    main()
    logging.info("End Training")
