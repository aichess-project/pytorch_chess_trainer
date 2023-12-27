from libs.log_lib import setup_logging
from config.trainer_config import save_trainer_config_to_yaml, Trainer_Config
from config.net_config import save_net_config_to_yaml, Net_Config
from config.dl_config import save_dl_config_to_yaml, DL_Config
import logging

def main():
    t_conf = Trainer_Config()
    #save_trainer_config_to_yaml(t_conf, "trainer_config.yaml")
    n_conf = Net_Config()
    save_net_config_to_yaml(n_conf, "net_config.yaml")
    dl_conf = DL_Config()
    #save_dl_config_to_yaml(dl_conf, "dl_config.yaml")

if __name__ == '__main__':
    setup_logging()
    logging.info("Start Training")
    main()
    logging.info("End Training")