import logging.config
import colorlog
from src.settings import SETTINGS

root_logger = logging.getLogger()
root_logger.setLevel(SETTINGS.LOG_LEVEL)

for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

ch = logging.StreamHandler()
ch.setLevel(SETTINGS.LOG_LEVEL)

formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s%(reset)s : %(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
)

ch.setFormatter(formatter)
root_logger.addHandler(ch)
