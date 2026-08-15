# -*- coding: utf-8 -*-
"""模型保存与加载模块"""

import os
import json
import torch
import yaml
from ..utils.logger import get_logger

logger = get_logger(__name__)


def save_best_model(model, optimizer, epoch, best_metric, config, path, scheduler=None):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_f1': best_metric,
        'config': config,
    }
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(state, path)
    logger.info(f"已保存最佳模型到 {path} (epoch={epoch}, best_f1={best_metric:.4f})")


def save_checkpoint(model, optimizer, epoch, path, scheduler=None, extra=None):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, path)
    logger.info(f"已保存检查点 {path} (epoch={epoch})")


def save_config(config, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    logger.info(f"已保存配置到 {path}")


def load_checkpoint(path, model=None, optimizer=None, scheduler=None, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(path, map_location=device)
    epoch = state.get('epoch', 0)
    if model is not None and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'], strict=True)
        logger.info(f"已加载模型权重 (epoch={epoch})")
    if optimizer is not None and 'optimizer_state_dict' in state:
        optimizer.load_state_dict(state['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in state:
        scheduler.load_state_dict(state['scheduler_state_dict'])
    return state, epoch


def load_config_from_checkpoint(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    return state.get('config', {})
