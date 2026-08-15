# -*- coding: utf-8 -*-
"""日志工具模块"""


import logging
import os
import yaml

_log_level = None

def set_log_level(level):
    global _log_level
    if isinstance(level, str):
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
        }
        _log_level = level_map.get(level.upper(), logging.INFO)
    else:
        _log_level = level

def get_log_level():
    global _log_level
    if _log_level is None:
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'configs', 'train_config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f)
                    level_str = cfg.get('log_level', 'INFO')
                    set_log_level(level_str)
            else:
                _log_level = logging.INFO
        except:
            _log_level = logging.INFO
    return _log_level


def get_logger(name, log_file=None, level=None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    if level is None:
        level = get_log_level()
    logger.setLevel(level)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(level)
    logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
