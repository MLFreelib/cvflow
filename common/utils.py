import argparse
import logging
import logging.config


class Logger:
    r""" Logger class. If the logger cannot be built, nothing will be output.

        :param conf_path: str
                    log configuration path.
        :param logger_name: str
    """

    def __init__(self):
        self.__conf_dict = self.__get_conf_dict()
        self.__logger = None

    def write(self, msg: str, lvl: str = "INFO"):
        r""" Write message to logger.
            :param msg: str
                    message.
            :param lvl: str
                    ["INFO", "DEBUG", "CRITICAL", "WARN"]
        """
        if self.__logger is not None:
            if lvl == "INFO":
                self.__logger.info(msg)
            elif lvl == "DEBUG":
                self.__logger.debug(msg)
            elif lvl == "CRITICAL":
                self.__logger.critical(msg)
            elif lvl == "WARN":
                self.__logger.warning(msg)

    def add_logger(self, name: str, options: dict):
        self.__conf_dict['loggers'][name] = options

    def get_handlers(self):
        return list(self.__conf_dict['handlers'].keys())

    def add_handler(self, handler_name: str, options: dict):
        self.__logger['handlers'][handler_name] = options

    def compile_logger(self, logger_name: str):
        logging.config.dictConfig(self.__conf_dict)
        self.__logger = logging.getLogger(logger_name)

    def __get_conf_dict(self):
        return {
            'version': 1,
            "handlers": {
                "fileHandler": {
                    "class": "logging.FileHandler",
                    "formatter": "cvflow_formatter",
                    "filename": "conf.log"
                },
                "consoleHandler": {
                    "class": "logging.StreamHandler",
                    "formatter": "cvflow_formatter"
                }
            },
            "loggers": dict(),
            "formatters": {
                "cvflow_formatter": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            }
        }


argparser = argparse.ArgumentParser()

argparser.add_argument('--usbcam', required=False)
argparser.add_argument('--videofile', required=False)
argparser.add_argument('-c', '--confidence', required=False)
argparser.add_argument('-f', '--font', required=False)
argparser.add_argument('--tsize', required=False)
argparser.add_argument('-d', '--device', required=False)

args = vars(argparser.parse_args())