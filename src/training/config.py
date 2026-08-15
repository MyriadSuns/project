# -*- coding: utf-8 -*-
"""训练与模型配置加载模块"""

import os
import yaml


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_train_config(config_path=None):
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if config_path is None:
        config_path = os.path.join(base, 'configs', 'train_config.yaml')
    cfg = load_yaml(config_path)
    for key in ['data_root_weibo', 'data_root_tiktok', 'bert_path', 'llm_path', 'checkpoint_dir', 'result_dir']:
        if key in cfg and not os.path.isabs(cfg[key]):
            cfg[key] = os.path.join(base, cfg[key])
    return cfg


def get_model_config(config_path=None):
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if config_path is None:
        config_path = os.path.join(base, 'configs', 'model_config.yaml')
    return load_yaml(config_path)
