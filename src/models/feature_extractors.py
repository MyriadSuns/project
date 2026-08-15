# -*- coding: utf-8 -*-
"""多模态特征提取器模块，用于从不同模态的输入中提取特征："""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class BaseFeatureExtractor(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _ensure_device(self, tensor):
        return tensor.to(self.device)


class BERTTextExtractor(BaseFeatureExtractor):
    def __init__(self, model_path=None, proj_dim=128, device=None):
        super().__init__(device=device)
        from transformers import BertModel, BertTokenizer
        
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, 'models', 'bert-base-chinese')
        
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.bert = BertModel.from_pretrained(model_path)
        self.bert = self._ensure_device(self.bert)
        # 冻结所有 BERT 层，仅解冻最后 2 层用于下游微调
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
        
        self.raw_dim = 768
        self.out_dim = proj_dim
        
        self.proj = nn.Linear(self.raw_dim, proj_dim)
        self.proj = self._ensure_device(self.proj)
    
    def forward(self, input_ids, attention_mask, return_raw=False):
        input_ids = self._ensure_device(input_ids)
        attention_mask = self._ensure_device(attention_mask)
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = outputs.last_hidden_state[:, 0, :]  
        
        if return_raw:
            return cls_vec
        
        proj_vec = self.proj(cls_vec)  
        return proj_vec
    
    def encode_text(self, text, max_length=512):
        if isinstance(text, str):
            text = [text]
        
        enc = self.tokenizer(
            text,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        enc = {k: self._ensure_device(v) for k, v in enc.items()}
        
        with torch.no_grad():
            feats = self.forward(enc['input_ids'], enc['attention_mask'])
        
        return feats


class ResNet50VisualExtractor(BaseFeatureExtractor):
    def __init__(self, pretrained=True, frame_proj_dim=256, device=None):
        super().__init__(device=device)
        from torchvision.models import resnet50, ResNet50_Weights
        
        if pretrained:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            model = resnet50(weights=None)

        model.fc = nn.Identity()
        self.backbone = self._ensure_device(model)
        self.backbone.eval()

        self.raw_dim = 2048
        self.frame_proj_dim = frame_proj_dim
        
        self.frame_proj = nn.Linear(self.raw_dim, frame_proj_dim)
        self.frame_proj = self._ensure_device(self.frame_proj)
        
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor() 
        ])
    
    def _compute_clarity_score(self, frame):
        try:
            if isinstance(frame, np.ndarray):
                
                frame = np.ascontiguousarray(frame)
                if len(frame.shape) == 3:
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    except:
                        try:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        except:
                            return 0.0
                else:
                    gray = frame
            else:
                if hasattr(frame, 'convert'):
                    gray = np.array(frame.convert('L'))
                else:
                    try:
                        gray = np.array(frame)
                        if isinstance(gray, np.ndarray):
                            gray = np.ascontiguousarray(gray)
                            if len(gray.shape) == 3:
                                try:
                                    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
                                except:
                                    try:
                                        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                                    except:
                                        return 0.0
                        else:
                            return 0.0
                    except:
                        return 0.0
            
            if not isinstance(gray, np.ndarray) or len(gray.shape) != 2:
                return 0.0
            
            # 计算清晰度评分：S=1/(H×W)∑I(i,j)²
            H, W = gray.shape
            total_pixels = H * W
            squared_sum = np.sum(np.square(gray.astype(np.float64)))
            S = squared_sum / total_pixels
            return S
        except Exception as e:
            print(f"计算清晰度评分时出错: {e}")
            return 0.0
    
    def _extract_video_keyframes(self, video_path, fps_sample=1.0, 
                               clarity_threshold=0.3, max_frames=32):
        frames = []
        
        if not os.path.exists(video_path):
            return frames
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return frames
        
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(video_fps / fps_sample))
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                score = self._compute_clarity_score(frame_rgb)
                if score >= clarity_threshold:
                    frames.append(Image.fromarray(frame_rgb))
            
            frame_idx += 1
        
        cap.release()
        
        if not frames and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
                cap.release()
        
        if len(frames) > max_frames:
            step = len(frames) / max_frames
            indices = [int(i * step) for i in range(max_frames)]
            frames = [frames[i] for i in indices]
        
        return frames
    
    def forward(self, x):
        x = self._ensure_device(x)
        feats = self.backbone(x)
        return feats
    
    def _preprocess_image(self, image):
        if isinstance(image, Image.Image):
            image = self.transform(image).unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
        return self._ensure_device(image)
    
    def encode_image(self, image):
        processed_image = self._preprocess_image(image)
        
        with torch.no_grad():
            raw_feats = self.forward(processed_image)  
            feats_256 = self.frame_proj(raw_feats)  
        
        return feats_256
    
    def encode_video_frames(self, list_of_frames):
        if not list_of_frames:
            return torch.zeros(1, self.frame_proj_dim, device=self.device)
        
        processed_frames = []
        for frame in list_of_frames:
            if isinstance(frame, Image.Image):
                processed = self.transform(frame).unsqueeze(0)
            else:
                processed = frame.unsqueeze(0) if frame.dim() == 3 else frame
            processed_frames.append(processed)
        
        batch_frames = torch.cat(processed_frames, dim=0)
        batch_frames = self._ensure_device(batch_frames)
        
        with torch.no_grad():
            raw_feats = self.forward(batch_frames)  
            feats_256 = self.frame_proj(raw_feats)  
            video_feat = feats_256.mean(dim=0, keepdim=True)  
        
        return video_feat


class MultimodalFeatureExtractors:
    """多模态特征提取器统一接口"""
    def __init__(self, bert_path=None):
        self.text_extractor = BERTTextExtractor(model_path=bert_path)
        self.visual_extractor = ResNet50VisualExtractor()
    
    def text_features(self, text, max_length=512):
        return self.text_extractor.encode_text(text, max_length=max_length)
    
    def image_features(self, image):
        return self.visual_extractor.encode_image(image)
    
    def video_features_from_frames(self, list_of_frames):
        return self.visual_extractor.encode_video_frames(list_of_frames)


def get_bert_extractor(model_path=None, device=None):
    return BERTTextExtractor(model_path=model_path, device=device)


def get_resnet50_extractor(pretrained=True, device=None):
    return ResNet50VisualExtractor(pretrained=pretrained, device=device)
