import logging
from torch.optim import SGD, Adam, RMSprop

logger = logging.getLogger("HSIC")

key2opt = {
    "SGD": SGD,
    "Adam": Adam,
    "RMSprop": RMSprop,
}

def get_optimizer(cfg):
    if cfg["Train"]["optimizer"] is None:
        logger.info("Using SGD optimizer")
        return SGD

    else:
        opt_name = cfg["Train"]["optimizer"]["optimizer_detail"]["optimizer_name"]
        if opt_name not in key2opt:
            raise NotImplementedError("Optimizer {} not implemented".format(opt_name))

        logger.info("Using {} optimizer".format(opt_name))
        return key2opt[opt_name]
