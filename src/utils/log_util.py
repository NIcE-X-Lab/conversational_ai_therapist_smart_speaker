# This script provides a utility for logging with colored output and file logging.
# It ensures that a log directory exists, and provides a get_logger function to create a logger with both stream and file handlers.
# The coloredlogs package is used to enhance the readability of logs in the terminal.

import os
import datetime
import coloredlogs
import logging
import sys
from src.utils.config_loader import LOG_DIR
# Ensure the log directory exists before logging
# Use umask to set directory permissions to 0777 for compatibility
if not os.path.exists(LOG_DIR):
    original_umask = os.umask(0)  # Set umask to 0 to allow full permissions
    os.makedirs(LOG_DIR, 0o777)  # Create the log directory with 0777 permissions
    os.umask(original_umask)  # Restore the original umask
    
_GLOBAL_LOG_FILE = os.environ.get("LOG_FILE")  # 可通过入口统一指定
_GLOBAL_FILE_HANDLER = None

def _ensure_global_file_handler():
    '''
    Ensure the global file handler is created.
    '''
    global _GLOBAL_FILE_HANDLER, _GLOBAL_LOG_FILE
    if _GLOBAL_FILE_HANDLER is None:
        if not _GLOBAL_LOG_FILE:
            _GLOBAL_LOG_FILE = os.path.join(
                LOG_DIR,
                datetime.datetime.now().strftime("output_%y%m%d_%H%M%S.log")
            )
        fh = logging.FileHandler(_GLOBAL_LOG_FILE)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s'))
        _GLOBAL_FILE_HANDLER = fh
    return _GLOBAL_FILE_HANDLER

def get_logger(name, file=None, file_handler=None):
    """
    Create and return a logger with both stream and file handlers.
    If file_handler is provided, it will be used for file logging.
    Otherwise, a new FileHandler will be created using the given file path or a timestamped default.
    Coloredlogs is used to enhance terminal output readability.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # file 仍写 DEBUG

    # 控制台日志级别来自环境变量，默认 DEBUG
    console_level_name = os.environ.get("CONSOLE_LOG_LEVEL", "INFO").upper()
    console_level = getattr(logging, console_level_name, logging.INFO)

    # 避免重复添加 handler
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        c_handler = logging.StreamHandler()  # 默认 stderr
        c_handler.setLevel(console_level)
        logger.addHandler(c_handler)

    # 文件 handler 仍为 DEBUG 级别
    if file_handler is not None:
        f_handler = file_handler
    else:
        if file is not None:
            f_handler = logging.FileHandler(file)
            f_handler.setLevel(logging.DEBUG)
            f_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s'))
        else:
            f_handler = _ensure_global_file_handler()

    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        logger.addHandler(f_handler)

    # coloredlogs 只影响控制台显示，跟随 console_level
    coloredlogs.install(level=console_level, logger=logger)

    return logger


