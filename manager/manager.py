from libs.log_lib import setup_logging
from trainer.trainer import Chess_Trainer
from data_loader.base_loader import Base_Dataset
#from config.trainer_config import load_trainer_config_from_yaml
from config.trainer_configlist import load_trainer_config_list_from_yaml, config_iterator
from config.net_config import get_net, load_net_config_from_yaml
from config.dl_config import get_dl, load_dl_config_from_yaml
from config.score import load_score_from_yaml, save_score_to_yaml
from config.conv_config import get_converter, load_conv_config_from_yaml
from trainer.trainer import Chess_Trainer
import logging, sys


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

def create_datasets(dl_config_file, training_steps, converter):
    datasets = {}
    dl_config = load_dl_config_from_yaml(dl_config_file)
    logging.info(f"DL Config: {dl_config}")
    for step in training_steps:
      dl_class = get_dl(dl_config)
      dl_class.init(dl_config, converter, step)
      datasets[step] = dl_class
      logging.info(f"Length Dataset {step}: {len(datasets[step])}")
    return datasets

def create_trainer(trainer_config, data_sets, net):
    trainer = Chess_Trainer(trainer_config, data_sets, net)
    return trainer    

def main(config_file = "trainer_config_list.yaml"):
    trainer_config_list = load_trainer_config_list_from_yaml(config_file)
    score = load_score_from_yaml(trainer_config_list.score_data)
    logging.info(f"Score: {score}")
    converter = create_converter(trainer_config_list.conv_config)
    net = create_net(trainer_config_list.net_config)
    data_sets = create_datasets(trainer_config_list.dl_config, trainer_config_list.training_steps, converter)
    for trainer_config in config_iterator(trainer_config_list):
        logging.info(f"Run: {score.next_run} Trainer Config: {trainer_config}")    
        trainer = create_trainer(trainer_config, data_sets, net)
        score = trainer.do_training(score, num_epochs=trainer_config.epochs)
    save_score_to_yaml(score, trainer_config_list.score_data)

if __name__ == "__main__":
    setup_logging()
    # Check if at least one argument is provided
    if len(sys.argv) < 2:
        print("Usage: python3 manager.py <config_file>")
        sys.exit(1)

    # Access the parameter passed
    config_file = sys.argv[1]
    logging.info(f"Start Training {config_file}")
    main(config_file)
    logging.info("End Training")
