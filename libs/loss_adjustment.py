import math, logging
def adjust_loss(loss, step, trainer_config):
    logging.debug(f"adjust_loss: {loss} {math.sqrt(loss)} {step} {trainer_config.reduction} {trainer_config.criterion}")
    if step == "test":
        return loss
    if trainer_config.criterion == "MSELoss":
        loss = math.sqrt(loss)
    return loss