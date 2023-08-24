import argparse
import logging
import logging.config
import os


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


class ConfigParser:
    def __init__(self, path: str):
        self.__path = path
        self.__config = dict()

    def read(self):
        import json
        with open(self.__path) as f:
            d = json.load(f)
            self.__config = d

    def get_value(self, key: str):
        return self.__config.get(key)


argparser = argparse.ArgumentParser()

argparser.add_argument('--usbcam', required=False)
argparser.add_argument('--videofile', required=False)
argparser.add_argument('--images', required=False)
argparser.add_argument('-c', '--confidence', required=False)
argparser.add_argument('-f', '--font', required=False)
argparser.add_argument('--tsize', required=False)
argparser.add_argument('--fsize', required=False)
argparser.add_argument('-d', '--device', required=False)
argparser.add_argument('-l', '--line', required=False)
argparser.add_argument('--data', required=False)
argparser.add_argument('-n', '--num', required=False, help='Number of tracking objects')
argparser.add_argument('-d', '--destination', required=False)

args = vars(argparser.parse_args())


def get_video_file_srcs():
    return get_src('videofile')


def get_cam_srcs():
    return get_src('usbcam')


def get_img_srcs():
    return get_src('images')


def get_data_srcs():
    data_path = args['data']
    assert os.path.exists(data_path)
    return data_path


def get_src(reader_name: str):
    readers = list()
    srcs = args[reader_name]
    if srcs is not None:
        readers = srcs.split(',')
    return readers


def get_confidence():
    try:
        return float(args['confidence'])
    except ValueError:
        return .0
    except AttributeError:
        return .0
    except TypeError:
        return .0


def get_font():
    return args['font']


def get_tsize():
    try:
        return [int(v) for v in args['tsize'].split(',')]
    except ValueError:
        return 2, 2
    except AttributeError:
        return 2, 2
    except TypeError:
        return 2, 2


def get_fsize():
    try:
        return [int(v) for v in args['fsize'].split(',')]
    except ValueError:
        return None
    except AttributeError:
        return 640, 960
    except TypeError:
        return 640, 960


def get_device():
    return 'cpu' if args['device'] is None else args['device']


def get_line():
    try:
        return [int(v) for v in args['line'].split(',')]
    except ValueError:
        return None
    except AttributeError:
        return 0, 0, 0, 0
    except TypeError:
        return 0, 0, 0, 0
