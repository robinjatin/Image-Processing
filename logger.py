import logging
import os
import sys
# from logging.handlers import TimedRotatingFileHandler

__FORMATTER = logging.Formatter("%(asctime)s — %(thread)d - %(name)s:%(lineno)s — %(levelname)s — %(message)s")


def __get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(__FORMATTER)
    return console_handler


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    profile = os.environ.get('LOG_LEVEL', None)
    if profile == "debug":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.addHandler(__get_console_handler())
    # logger.addHandler(get_file_handler())
    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False
    return logger
