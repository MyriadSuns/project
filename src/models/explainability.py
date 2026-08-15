# -*- coding: utf-8 -*-
"""模型可解释性分析模块"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image


class TextSHAPAnalyzer:
    """文本SHAP分析"""
    
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.max_length = 512
    
    def analyze(self, text: str):
        """分析文本中每个token的重要性"""
        import numpy as np
        
        was_training = self.model.training
        bert_was_training = self.model.text_encoder.bert.training
        
        try:
            self.model.eval()
            self.model.zero_grad()

            enc = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
            input_ids = enc['input_ids'].to(self.device)
            attention_mask = enc['attention_mask'].to(self.device)
            
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            token_texts = []
            for t in input_ids[0].tolist():
                try:
                    token_texts.append(self.tokenizer.decode([t]))
                except:
                    token_texts.append('')
            
            word_embeddings = self.model.text_encoder.bert.embeddings.word_embeddings
            input_embeds = word_embeddings(input_ids).clone().detach().requires_grad_(True)
            
            self.model.text_encoder.bert.train()
            
            outputs = self.model.text_encoder.bert(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            cls_vec = outputs.last_hidden_state[:, 0, :]
            text_feat = self.model.text_encoder.proj(cls_vec)
            
            text_expert_input = self.model.fusion_model.text_proj_to_expert(text_feat)
            text_score = self.model.fusion_model.moe_detector.text_expert(text_expert_input)
            
            text_score.sum().backward()
            
            if input_embeds.grad is None:
                return {
                    'tokens': token_texts,
                    'importance_scores': [0.0] * len(token_texts),
                    'top_important_tokens': [],
                    'method': 'gradient',
                    'total_tokens': len(tokens),
                    'error': '梯度计算失败',
                    'debug': {
                        'input_embeds_grad_none': True,
                        'model_training': was_training,
                        'bert_training': bert_was_training
                    }
                }
            
            gradients = input_embeds.grad.data
            importance_scores = gradients.abs().sum(dim=-1).squeeze(0).cpu().numpy()
            
            scores_array = np.array(importance_scores)
            max_score = scores_array.max()
            min_score = scores_array.min()
            
            if max_score > 0:
                scores_array = scores_array / max_score
            else:
                scores_array = np.zeros_like(scores_array)
            
            importance_scores_normalized = scores_array.tolist()
            
            top_indices = np.argsort(scores_array)[::-1][:10]
            top_tokens = []
            for idx in top_indices:
                if scores_array[idx] > 0.01 and tokens[idx] not in ['[CLS]', '[SEP]', '[PAD]']:
                    top_tokens.append({
                        'token': token_texts[idx],
                        'importance': float(scores_array[idx])
                    })
            
            return {
                'tokens': token_texts,
                'importance_scores': importance_scores_normalized,
                'top_important_tokens': top_tokens,
                'method': 'gradient',
                'total_tokens': len(tokens),
                'max_raw_score': float(max_score) if max_score > 0 else 0.0,
                'min_raw_score': float(min_score),
                'debug': {
                    'max_raw_score': float(max_score) if max_score > 0 else 0.0,
                    'min_raw_score': float(min_score),
                    'non_zero_count': int((scores_array > 0).sum()),
                    'model_training': was_training,
                    'bert_training': bert_was_training
                }
            }
            
        finally:
            self.model.zero_grad()
            if not bert_was_training:
                self.model.text_encoder.bert.eval()
            if was_training:
                self.model.train()


class VisualSHAPAnalyzer:
    """视觉SHAP分析"""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
    
    def analyze(self, image_tensor):
        """分析图像中每个区域的重要性"""
        import numpy as np
        
        if image_tensor is None:
            return None
        
        was_training = self.model.image_encoder.backbone.training
        
        try:
            self.model.eval()
            self.model.zero_grad()
            
            image_tensor = image_tensor.clone().detach().requires_grad_(True)
            
            self.model.image_encoder.backbone.train()
            
            visual_feat = self.model.image_encoder.backbone(image_tensor)
            
            visual_expert_input = self.model.fusion_model.visual_proj_to_expert(visual_feat)
            visual_expert_input = visual_expert_input.unsqueeze(1)
            visual_score = self.model.fusion_model.moe_detector.visual_expert(visual_expert_input)
            
            visual_score.sum().backward()
            
            if image_tensor.grad is None:
                return {
                    'heatmap_available': False,
                    'error': '梯度计算失败',
                    'method': 'gradient'
                }
            
            gradients = image_tensor.grad.data
            
            if gradients.abs().sum() == 0:
                return {
                    'heatmap_available': False,
                    'error': '梯度为零',
                    'method': 'gradient'
                }
            
            gradients = gradients.abs().mean(dim=1, keepdim=True)
            
            from torch.nn import functional as F
            heatmap = F.interpolate(gradients, size=(7, 7), mode='bilinear', align_corners=False)
            heatmap = heatmap.squeeze().cpu().numpy()
            
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            
            return {
                'heatmap_available': True,
                'shape': [7, 7],
                'max_importance': float(heatmap.max()),
                'mean_importance': float(heatmap.mean()),
                'patch_importance': heatmap.tolist(),
                'method': 'gradient'
            }
        finally:
            self.model.zero_grad()
            if not was_training:
                self.model.image_encoder.backbone.eval()


def run_shap_analysis(model, tokenizer, text, image_tensors=None, video_frames=None, device=None):
    """运行SHAP分析"""
    result = {"enabled": True, "text_shap": None, "visual_shap": None, "images_shap": None, "video_shap": None, "error": None}
    
    try:
        text_analyzer = TextSHAPAnalyzer(model, tokenizer, device)
        result["text_shap"] = text_analyzer.analyze(text)
    except Exception as e:
        result["text_shap"] = {"error": str(e)}
    
    if image_tensors is not None and len(image_tensors) > 0:
        try:
            visual_analyzer = VisualSHAPAnalyzer(model, device)
            result["visual_shap"] = visual_analyzer.analyze(image_tensors[0])
            
            images_data = []
            for i, img_tensor in enumerate(image_tensors[:3]):
                img_result = visual_analyzer.analyze(img_tensor)
                images_data.append({
                    'image_index': i,
                    'heatmap_available': img_result.get('heatmap_available', False),
                    'patch_importance': img_result.get('patch_importance', []),
                    'max_importance': img_result.get('max_importance', 0.0),
                    'mean_importance': img_result.get('mean_importance', 0.0)
                })
            result["images_shap"] = {
                "images": images_data,
                "total_images": len(image_tensors),
                "analyzed_images": len(images_data)
            }
        except Exception as e:
            result["visual_shap"] = {"error": str(e)}
    
    if video_frames is not None and len(video_frames) > 0:
        try:
            visual_analyzer = VisualSHAPAnalyzer(model, device)
            video_results = []
            import base64
            from io import BytesIO
            
            for idx, frame in enumerate(video_frames[:5]):
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                frame_tensor = transform(frame).unsqueeze(0).to(device)
                frame_result = visual_analyzer.analyze(frame_tensor)
                
                buffer = BytesIO()
                frame.save(buffer, format='JPEG')
                frame_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                frame_data = {
                    'frame_index': idx,
                    'frame_base64': frame_base64,
                    'heatmap_available': frame_result.get('heatmap_available', False),
                    'patch_importance': frame_result.get('patch_importance', []),
                    'max_importance': frame_result.get('max_importance', 0.0),
                    'mean_importance': frame_result.get('mean_importance', 0.0)
                }
                video_results.append(frame_data)
            
            result["video_shap"] = {
                "frames": video_results,
                "total_frames": len(video_frames),
                "analyzed_frames": len(video_results)
            }
        except Exception as e:
            result["video_shap"] = {"error": str(e)}
    
    return result
