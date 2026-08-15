# -*- coding: utf-8 -*-
"""跨模态注意力机制模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    def __init__(self, dim1, dim2, scale=None):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.W = nn.Linear(dim1, dim2, bias=False)
        if scale is None:
            scale = (dim2 ** -0.5)
        self.scale = scale
        
    def forward(self, feat1, feat2):
        feat1_proj = self.W(feat1)  
        S = torch.matmul(feat1_proj, feat2.transpose(1, 2)) * self.scale
        A_12 = F.softmax(S, dim=-1)  
        A_21 = F.softmax(S.transpose(1, 2), dim=-1)  
        enhanced1 = torch.matmul(A_12, feat2)  
        enhanced2 = torch.matmul(A_21, feat1_proj)  

        return enhanced1, enhanced2


class TextVisualAttention(CrossModalAttention):
    def __init__(self, text_dim=768, visual_dim=2048, scale=None):
        super().__init__(text_dim, visual_dim, scale)


class TextEvidenceAttention(CrossModalAttention):
    def __init__(self, text_dim=768, evidence_dim=256, scale=None):
        super().__init__(text_dim, evidence_dim, scale)
    
    def forward(self, text_feat, evidence_feat):
        if evidence_feat.dim() == 2:
            evidence_feat = evidence_feat.unsqueeze(1)  
        text_enhanced, evidence_enhanced = super().forward(text_feat, evidence_feat)
        if evidence_enhanced.size(1) == 1:
            evidence_enhanced = evidence_enhanced.squeeze(1)
        
        return text_enhanced, evidence_enhanced


class VisualEvidenceAttention(CrossModalAttention):
    def __init__(self, visual_dim=2048, evidence_dim=256, scale=None):
        super().__init__(visual_dim, evidence_dim, scale)
    
    def forward(self, visual_feat, evidence_feat):
        if evidence_feat.dim() == 2:
            evidence_feat = evidence_feat.unsqueeze(1)  
        visual_enhanced, evidence_enhanced = super().forward(visual_feat, evidence_feat)
        if evidence_enhanced.size(1) == 1:
            evidence_enhanced = evidence_enhanced.squeeze(1)
        
        return visual_enhanced, evidence_enhanced


class CrossModalAlignment(nn.Module):
    """完整的跨模态对齐模块
    实现了三组双向注意力 + 全局融合 + 残差连接，用于对齐文本、视觉和证据三个模态的特征。
    """
    
    def __init__(
        self,
        text_dim=768,
        visual_dim=2048,
        evidence_dim=256,
        text_seq_len=128,
        visual_patches=49,  
    ):
        super().__init__()
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.evidence_dim = evidence_dim

        self.text_visual_attn = TextVisualAttention(text_dim, visual_dim)
        self.text_evidence_attn = TextEvidenceAttention(text_dim, evidence_dim)
        self.visual_evidence_attn = VisualEvidenceAttention(visual_dim, evidence_dim)

        self.text_proj = nn.Linear(visual_dim, text_dim) 
        self.text_evidence_proj = nn.Linear(evidence_dim, text_dim)  

        self.visual_proj = nn.Linear(visual_dim, visual_dim)  
        self.visual_evidence_proj = nn.Linear(evidence_dim, visual_dim) 

        self.evidence_text_proj = nn.Linear(evidence_dim, evidence_dim) 
        self.evidence_visual_proj = nn.Linear(evidence_dim, evidence_dim) 
        
        self.text_alpha = nn.Parameter(torch.tensor([0.5, 0.5]))  
        self.visual_alpha = nn.Parameter(torch.tensor([0.5, 0.5]))  
        self.evidence_alpha = nn.Parameter(torch.tensor([0.5, 0.5]))  
        
    def forward(self, text_feat, visual_feat, evidence_feat, 
                has_visual=True, has_evidence=True):   
        if visual_feat.dim() == 2:
            visual_feat = visual_feat.unsqueeze(1)
        
        batch_size = text_feat.size(0)
        device = text_feat.device
        
        text_from_visual = torch.zeros(batch_size, text_feat.size(1), self.text_dim, device=device)
        text_from_evidence = torch.zeros(batch_size, text_feat.size(1), self.text_dim, device=device)
        visual_from_text = torch.zeros_like(visual_feat)
        visual_from_evidence = torch.zeros_like(visual_feat)
        evidence_from_text = torch.zeros(batch_size, self.evidence_dim, device=device)
        evidence_from_visual = torch.zeros(batch_size, self.evidence_dim, device=device)
        
        if has_visual:
            text_enhanced_v, visual_enhanced_t = self.text_visual_attn(text_feat, visual_feat)
            text_from_visual = self.text_proj(text_enhanced_v)
            visual_from_text = self.visual_proj(visual_enhanced_t)
            
            if has_evidence:
                visual_enhanced_e, evidence_enhanced_v = self.visual_evidence_attn(visual_feat, evidence_feat)
                visual_from_evidence = self.visual_evidence_proj(visual_enhanced_e)
                evidence_from_visual = self.evidence_visual_proj(evidence_enhanced_v)
        
        if has_evidence:
            text_enhanced_e, evidence_enhanced_t = self.text_evidence_attn(text_feat, evidence_feat)
            text_from_evidence = self.text_evidence_proj(text_enhanced_e)
            evidence_from_text = self.evidence_text_proj(evidence_enhanced_t)
        
        text_fused = text_feat + self.text_alpha[0] * text_from_visual + self.text_alpha[1] * text_from_evidence
        visual_fused = visual_feat + self.visual_alpha[0] * visual_from_text + self.visual_alpha[1] * visual_from_evidence
        evidence_fused = evidence_feat + self.evidence_alpha[0] * evidence_from_text + self.evidence_alpha[1] * evidence_from_visual
        
        text_fused = F.layer_norm(text_fused, text_fused.shape[-1:])
        visual_fused = F.layer_norm(visual_fused, visual_fused.shape[-1:])
        evidence_fused = F.layer_norm(evidence_fused, evidence_fused.shape[-1:])
        
        return text_fused, visual_fused, evidence_fused
