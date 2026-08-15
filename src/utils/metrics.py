# -*- coding: utf-8 -*-
"""评估指标模块"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    classification_report
)


def compute_metrics(preds, labels, scores=None, num_classes=2):
    if torch.is_tensor(preds):
        preds = preds.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if scores is not None and torch.is_tensor(scores):
        scores = scores.cpu().numpy()
    
    preds = np.asarray(preds).ravel()
    labels = np.asarray(labels).ravel()
    
    acc = accuracy_score(labels, preds)
    
    if num_classes == 2:
        p = precision_score(labels, preds, average='binary', zero_division=0)
        r = recall_score(labels, preds, average='binary', zero_division=0)
        f1 = f1_score(labels, preds, average='binary', zero_division=0)
    else:
        p = precision_score(labels, preds, average='weighted', zero_division=0)
        r = recall_score(labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    
    result = {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}

    if scores is not None:
        try:
            if num_classes == 2:
                auc = roc_auc_score(labels, scores)
            else:
                auc = roc_auc_score(labels, scores, multi_class='ovr', average='weighted')
            result['auc'] = auc
        except Exception:
            result['auc'] = 0.0
    
    return result


def classification_report_str(preds, labels, target_names=None):
    if torch.is_tensor(preds):
        preds = preds.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if target_names is None:
        target_names = ['real', 'fake']
    return classification_report(labels, preds, target_names=target_names)
