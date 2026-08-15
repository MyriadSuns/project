# -*- coding: utf-8 -*-
"""多模态融合模型模块"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_modal_attention import CrossModalAlignment
from .moe_model import MoEFakeNewsDetector


class MultimodalFusionModel(nn.Module):
    """完整的多模态虚假新闻检测模型"""
    def __init__(
        self,
        text_dim=768,
        visual_dim=2048,
        evidence_dim=256,
        text_seq_len=128,
        visual_patches=49,
        text_expert_dim=256,
        visual_expert_dim=512,
        evidence_expert_dim=256,
        dropout=0.3,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.evidence_dim = evidence_dim
        
        self.cross_modal_align = CrossModalAlignment(
            text_dim=text_dim,
            visual_dim=visual_dim,
            evidence_dim=256,
            text_seq_len=text_seq_len,
            visual_patches=visual_patches,
        )
        
        self.text_proj_to_expert = nn.Sequential(
            nn.Linear(text_dim, text_expert_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(text_expert_dim * 2, text_expert_dim),
        )
        self.visual_proj_to_expert = nn.Sequential(
            nn.Linear(visual_dim, visual_expert_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(visual_expert_dim * 2, visual_expert_dim),
        )
        self.evidence_proj_to_expert = nn.Sequential(
            nn.Linear(256, evidence_expert_dim * 2),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(evidence_expert_dim * 2, evidence_expert_dim),
        )
        
        self.moe_detector = MoEFakeNewsDetector(
            text_dim=text_expert_dim,
            visual_dim=visual_expert_dim,
            evidence_dim=evidence_expert_dim,
        )

    def forward(
        self,
        text_feat,
        visual_feat,
        evidence_feat,
        has_visual=True,
        has_evidence=True,
    ):
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)
        
        if visual_feat.dim() == 2:
            visual_feat = visual_feat.unsqueeze(1)
        
        text_aligned, visual_aligned, evidence_aligned = self.cross_modal_align(
            text_feat, visual_feat, evidence_feat, has_visual, has_evidence
        )
        
        if text_aligned.dim() == 3:
            text_pooled = text_aligned.mean(dim=1)
        else:
            text_pooled = text_aligned
        text_expert_input = self.text_proj_to_expert(text_pooled)
        
        visual_expert_input = self.visual_proj_to_expert(visual_aligned)
        
        evidence_expert_input = self.evidence_proj_to_expert(evidence_aligned)
        
        final_score, weights = self.moe_detector(
            text_expert_input,
            visual_expert_input,
            evidence_expert_input,
            has_text=True,
            has_visual=has_visual,
            has_evidence=has_evidence,
        )
        
        return final_score, weights


class MultimodalFusionWithExtractors(nn.Module):
    """带特征提取器的完整多模态融合模型"""
    def __init__(
        self,
        text_encoder,
        image_encoder,
        fusion_model,
        video_encoder=None,
        llm_wrapper=None,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.video_encoder = video_encoder
        self.llm_wrapper = llm_wrapper
        self.fusion_model = fusion_model

    def forward(
        self,
        input_ids,
        attention_mask,
        image,
        evidence_feat,
        video_path=None,
        text=None,
        has_image=True,
        has_video=None,
    ):
        if isinstance(has_image, torch.Tensor):
            has_image_bool = has_image.any().item()
        else:
            has_image_bool = bool(has_image) if has_image is not None else False
        
        if isinstance(has_video, torch.Tensor):
            has_video_bool = has_video.any().item()
        else:
            has_video_bool = bool(has_video) if has_video is not None else False
        
        has_visual = has_image_bool or has_video_bool
        has_evidence = evidence_feat is not None and not torch.all(evidence_feat == 0)
        
        text_feat = self.text_encoder(input_ids, attention_mask)
        text_feat = text_feat.unsqueeze(1)

        with torch.no_grad():
            image_feat_global = self.image_encoder(image)
            image_feat = image_feat_global.unsqueeze(1)
            
            if video_path and video_path != '' and self.video_encoder is not None:
                try:
                    if isinstance(video_path, list):
                        batch_size = image_feat.size(0)
                        video_features = []
                        
                        for i in range(batch_size):
                            try:
                                current_video_path = video_path[i] if i < len(video_path) else ''
                                
                                if not isinstance(current_video_path, (str, bytes, os.PathLike)):
                                    current_video_path = str(current_video_path) if current_video_path else ''
                                
                                if current_video_path and current_video_path != '':
                                    video_feat = self.video_encoder.encode_video_from_path(current_video_path)
                                    if video_feat is not None:
                                        video_features.append(video_feat)
                                    else:
                                        video_features.append(torch.zeros(1, 256, device=image_feat.device))
                                else:
                                    video_features.append(torch.zeros(1, 256, device=image_feat.device))
                            except Exception:
                                video_features.append(torch.zeros(1, 256, device=image_feat.device))
                        
                        if video_features:
                            video_feat_batch = torch.cat(video_features, dim=0)
                            video_feat_batch = video_feat_batch.unsqueeze(1)
                            video_proj = nn.Linear(256, 2048).to(image_feat.device)
                            video_feat_proj = video_proj(video_feat_batch)
                            try:
                                image_feat = torch.cat([image_feat, video_feat_proj], dim=1)
                            except Exception:
                                raise
                except Exception:
                    pass
        
        evidence_feat = evidence_feat.to(text_feat.device)
        
        final_score, weights = self.fusion_model(
            text_feat=text_feat,
            visual_feat=image_feat,
            evidence_feat=evidence_feat,
            has_visual=has_visual,
            has_evidence=has_evidence,
        )
        
        return final_score, weights
