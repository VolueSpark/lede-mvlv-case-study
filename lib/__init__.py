import logging, os, time, json
from datetime import datetime
from threading import Lock
import polars as pl
import pandas as pd

PATH = os.path.dirname(__file__)
LOG_PATH = PATH + '/../_logs/'

lock = Lock()


class bcolors:
    DEBUG = '\u001b[48;5;11m\u001b[38;5;0m'
    INFO = '\033[32m'
    NOTICE = '\033[94m'
    WARNING = '\033[93m'
    EXCEPTION = '\033[91m'
    ENDC = '\033[0m'


class Logging:

    def __init__(self, debug: bool = False, log_path: str = LOG_PATH, name: str='log'):
        self._debug = debug
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        logging.basicConfig(filename=os.path.join(log_path, f'{name}.log'), level=logging.INFO)



    def __thread_safe_print(self, msg: str):
        with lock:
            print(msg, flush=True)

    def __debug(self, msg:str, color: bcolors):
        self.__thread_safe_print(f"{color}{msg}{bcolors.ENDC}")


    def __log(self, msg:str, color: bcolors):
        self.__thread_safe_print(f"{color}{msg}{bcolors.ENDC}")
        if color in [bcolors.INFO, bcolors.NOTICE]:
            logging.info(msg)
        elif color == bcolors.WARNING:
            logging.warning(msg)
        elif color == bcolors.EXCEPTION:
            logging.exception(msg)


    def info(self, msg: str, color:bcolors=bcolors.INFO, ):
        self.__log(msg=msg, color=color)

    def notice(self, msg: str, color:bcolors=bcolors.NOTICE):
        self.__log(msg=msg,  color=color)

    def warning(self, msg: str, color:bcolors=bcolors.WARNING):
        self.__log(msg=msg,  color=color)

    def exception(self, msg: str, color:bcolors=bcolors.EXCEPTION):
        self.__log(msg=msg,  color=color)

    def debug(self,**kwargs):# df: pl.DataFrame, sort_by: str, n:int):
        if self._debug:
            color = kwargs['color'] if 'color' in kwargs else bcolors.DEBUG
            if 'df' in kwargs and type(kwargs['df']) == pl.DataFrame:
                with pl.Config() as cfg:
                    cfg.set_tbl_cols(-1)
                    cfg.set_tbl_width_chars(1000)
                    cfg.set_tbl_rows(-1)
                    if 'n' in kwargs:
                        print(kwargs['df'].head(n=kwargs['n']))
                    else:
                        print(kwargs['df'].head(n=100))
            elif 'df' in kwargs and type(kwargs['df']) == pd.DataFrame:
                pd.set_option('display.max_columns', None)
                pd.set_option("display.max_rows", 1000)
                pd.set_option('display.max_colwidth', None)
                print(kwargs['df'])

                pass
            elif 'msg' in kwargs and type(kwargs['msg']) == json:
                self.__debug(msg=json.dumps(json.loads(kwargs['msg']), indent=2), color=color)
            elif 'msg' in kwargs and type(kwargs['msg']) == dict:
                self.__debug(msg=json.dumps(kwargs['msg'], indent=2), color=color)
            elif 'msg' in kwargs and type(kwargs['msg']) == str:
                self.__debug(msg=kwargs['msg'], color=color)


def decorator_timer(func):
    def inner(*args, **kwargs):
        t0 = time.time()
        response = func(*args, **kwargs)
        dt = time.time()-t0
        logger.notice(f"[{datetime.now().isoformat()}] Profiler: {func.__module__+'.'+func.__qualname__} - {round(dt,2)} s")
        return response
    return inner


def sort(*args, **kwargs):
    if type(args[0]) == dict:
        if 'reverse' in kwargs:
            return dict(sorted(args[0].items(), key=lambda x:x[1], reverse=kwargs['reverse']))
        return dict(sorted(args[0].items(), key=lambda x:x[1], reverse=True))
    return args


logger = Logging(debug=True)
