from libs.log_lib import setup_logging
from trainer.trainer import Chess_Trainer
from config.trainer_config import load_trainer_config_from_yaml
from config.net_config import load_net_config_from_yaml
from config.dl_config import load_dl_config_from_yaml
import logging

def main():
    trainer_config = load_trainer_config_from_yaml("trainer_config.yaml")
    logging.info(f"Trainer Config: {trainer_config}")
    net_config = load_net_config_from_yaml("net_config.yaml")
    logging.info(f"Net Config: {net_config}")
    dl_config = load_dl_config_from_yaml("dl_config.yaml")
    logging.info(f"DL Config: {dl_config}")
    #trainer = Chess_Trainer(trainer_config, data_loaders, machine)

if __name__ == '__main__':
    setup_logging()
    logging.info("Start Training")
    main()
    logging.info("End Training")