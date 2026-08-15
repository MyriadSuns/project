# -*- coding: utf-8 -*-
"""混合专家模型模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class TextExpert(nn.Module):
    """文本专家 - 3层MLP + LayerNorm"""
    def __init__(self, input_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, text_feat):
        if text_feat.dim() > 2:
            text_feat = text_feat.mean(dim=1)
        return self.fc(text_feat)


class VisualExpert(nn.Module):
    """视觉专家 - 3层MLP + LayerNorm"""
    def __init__(self, input_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, visual_feat):
        if visual_feat.dim() > 2:
            visual_feat = visual_feat.mean(dim=1)
        return self.fc(visual_feat)


class EvidenceExpert(nn.Module):
    """证据专家 - 3层MLP + LayerNorm"""
    def __init__(self, evidence_feat_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(evidence_feat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, evidence_feat):
        if evidence_feat.dim() > 2:
            evidence_feat = evidence_feat.mean(dim=1)
        return self.fc(evidence_feat)


class GatingNetwork(nn.Module):
    def __init__(self, text_dim=128, visual_dim=256, evidence_dim=256, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        total_dim = text_dim + visual_dim + evidence_dim

        self.gate = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )

    def forward(self, text_feat, visual_feat, evidence_feat,
                has_text=True, has_visual=True, has_evidence=True):
        if text_feat.dim() > 2:
            text_feat = text_feat.mean(dim=1)
        if visual_feat.dim() > 2:
            visual_feat = visual_feat.mean(dim=1)
        if evidence_feat.dim() > 2:
            evidence_feat = evidence_feat.mean(dim=1)
        
        batch_size = text_feat.size(0)
        device = text_feat.device
        
        if isinstance(has_text, bool):
            has_text = torch.ones(batch_size, 1, device=device) if has_text else torch.zeros(batch_size, 1, device=device)
        if isinstance(has_visual, bool):
            has_visual = torch.ones(batch_size, 1, device=device) if has_visual else torch.zeros(batch_size, 1, device=device)
        if isinstance(has_evidence, bool):
            has_evidence = torch.ones(batch_size, 1, device=device) if has_evidence else torch.zeros(batch_size, 1, device=device)
        
        combined = torch.cat([text_feat, visual_feat, evidence_feat], dim=-1)
        gate_logits = self.gate(combined)
        
        availability = torch.cat([has_text, has_visual, has_evidence], dim=-1)
        masked_logits = gate_logits * availability
        weights = F.softmax(masked_logits / self.temperature, dim=-1)
        
        return weights


class MoEFakeNewsDetector(nn.Module):
    def __init__(self, text_dim=128, visual_dim=256, evidence_dim=256, temperature=1.0):
        super().__init__()
        self.text_expert = TextExpert(input_dim=text_dim)
        self.visual_expert = VisualExpert(input_dim=visual_dim)
        self.evidence_expert = EvidenceExpert(evidence_feat_dim=evidence_dim)
        self.gating_network = GatingNetwork(text_dim=text_dim, visual_dim=visual_dim, evidence_dim=evidence_dim, temperature=temperature)
        
    def forward(self, text_feat, visual_feat, evidence_feat, 
                has_text=True, has_visual=True, has_evidence=True):
        text_score = self.text_expert(text_feat)
        visual_score = self.visual_expert(visual_feat)
        evidence_score = self.evidence_expert(evidence_feat)
        
        weights = self.gating_network(text_feat, visual_feat, evidence_feat, 
                                       has_text, has_visual, has_evidence)
        
        scores = torch.cat([text_score, visual_score, evidence_score], dim=-1)
        final_score = (scores * weights).sum(dim=-1, keepdim=True)
        
        final_score = torch.sigmoid(final_score)
        
        return final_score, weights


def predict_with_threshold(model, text_feat, visual_feat, evidence_feat, 
                           has_text=True, has_visual=True, has_evidence=True, 
                           threshold=0.5):
    final_score, weights = model(text_feat, visual_feat, evidence_feat, 
                                  has_text, has_visual, has_evidence)
    final_score = final_score.squeeze(-1)
    predictions = (final_score >= threshold).long()
    low_confidence_mask = (final_score > 0.4) & (final_score < 0.6)
    
    return {
        'predictions': predictions,
        'scores': final_score,
        'weights': weights,
        'low_confidence': low_confidence_mask
    }
