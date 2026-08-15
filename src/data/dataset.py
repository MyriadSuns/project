# -*- coding: utf-8 -*-
"""多模态虚假新闻检测数据集类"""

import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from src.llm.evidence_extractor import proof_to_evidence_feat

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def get_image_transform(image_size=224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_train_transform(image_size=224):
    return T.Compose([
        T.RandomHorizontalFlip(p=0.3),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class MultimodalFakeNewsDataset(Dataset):
    def __init__(self, csv_path, data_root, tokenizer, image_transform=None, max_text_len=512,
                 evidence_dim=256, use_evidence=True, llm_wrapper=None, evidence_cache_path=None):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.image_transform = image_transform or get_image_transform()
        self.evidence_dim = evidence_dim
        self.use_evidence = use_evidence
        self.llm_wrapper = llm_wrapper
        self.samples = self._load_csv(csv_path)
        
        self.evidence_cache = None
        if evidence_cache_path and os.path.exists(evidence_cache_path):
            import json
            with open(evidence_cache_path, 'r', encoding='utf-8') as f:
                self.evidence_cache = json.load(f)
            import logging
            logging.getLogger(__name__).info(f"已加载证据缓存: {evidence_cache_path}, 共 {len(self.evidence_cache)} 条")

    def _load_csv(self, csv_path):
        samples = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append(row)
        return samples

    def __len__(self):
        return len(self.samples)

    def _process_image(self, image_path):
        import logging
        logger = logging.getLogger(__name__)
        
        has_image = 0.0
        if image_path:
            image_path = image_path.strip()
            if image_path:
                dataset_type = 'weibo' if 'weibo' in self.data_root.lower() else 'tiktok'
                image_dir = f'{dataset_type}_images'
                
                image_path = os.path.normpath(image_path)
                
                if os.path.basename(image_path) == image_path:
                    full_path = os.path.join(self.data_root, image_dir, image_path)
                else:
                    path_parts = image_path.split(os.sep)
                    first_dir = None
                    for part in path_parts:
                        if part and not part.startswith('.'):
                            first_dir = part
                            break
                    
                    if first_dir and first_dir in ['images', 'image', 'weibo_images', 'tiktok_images']:
                        corrected_path = os.path.join(image_dir, *path_parts[path_parts.index(first_dir)+1:])
                        full_path = os.path.join(self.data_root, corrected_path)
                    else:
                        full_path = os.path.join(self.data_root, image_path)
                
                full_path = os.path.normpath(full_path)
                
                if os.path.exists(full_path):
                    try:
                        img = Image.open(full_path).convert('RGB')
                        image_tensor = self.image_transform(img)
                        has_image = 1.0
                        logger.debug(f"成功加载图像: {full_path}")
                    except Exception as e:
                        logger.debug(f"加载图像失败: {full_path}, 错误: {e}")
                        image_tensor = torch.zeros(3, 224, 224)
                else:
                    logger.debug(f"图像路径不存在: {full_path}")
                    image_tensor = torch.zeros(3, 224, 224)
            else:
                image_tensor = torch.zeros(3, 224, 224)
        else:
            image_tensor = torch.zeros(3, 224, 224)
        
        return image_tensor, has_image

    def _process_video(self, video_path):
        import logging
        logger = logging.getLogger(__name__)
        
        has_video = 0.0
        video_full_path = ''
        if video_path:
            video_path = video_path.strip()
            if video_path:
                dataset_type = 'weibo' if 'weibo' in self.data_root.lower() else 'tiktok'
                video_dir = f'{dataset_type}_videos'
                
                video_path = os.path.normpath(video_path)
                
                if os.path.basename(video_path) == video_path:
                    full_path = os.path.join(self.data_root, video_dir, video_path)
                else:
                    path_parts = video_path.split(os.sep)
                    first_dir = None
                    for part in path_parts:
                        if part and not part.startswith('.'):
                            first_dir = part
                            break
                    
                    if first_dir and first_dir in ['video', 'videos', 'weibo_videos', 'tiktok_videos']:
                        corrected_path = os.path.join(video_dir, *path_parts[path_parts.index(first_dir)+1:])
                        full_path = os.path.join(self.data_root, corrected_path)
                    else:
                        full_path = os.path.join(self.data_root, video_path)
                
                full_path = os.path.normpath(full_path)
                
                if os.path.exists(full_path):
                    has_video = 1.0
                    video_full_path = full_path
                    logger.debug(f"成功找到视频: {full_path}")
                else:
                    logger.debug(f"视频路径不存在: {full_path}")
                    video_full_path = ''
        
        return video_full_path, has_video

    def __getitem__(self, idx):
        row = self.samples[idx]
        text = row.get('text', '')
        label = int(row.get('label', 0))
        image_path = row.get('image', '')
        video_path = row.get('video', '')

        encoding = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        image_tensor, has_image = self._process_image(image_path)
        video_full_path, has_video = self._process_video(video_path)

        out = {
            'input_ids': input_ids.cpu(),
            'attention_mask': attention_mask.cpu(),
            'image': image_tensor.cpu(), 
            'label': torch.tensor(label, dtype=torch.long).cpu(),
            'has_image': torch.tensor(has_image, dtype=torch.float32).cpu(),
            'has_video': torch.tensor(has_video, dtype=torch.float32).cpu(),
            'video_path': video_full_path,
            'text_raw': text,
            'id': row.get('id', ''),
        }

        if self.use_evidence:
            import logging
            logger = logging.getLogger(__name__)
            
            if self.evidence_cache is not None:
                cache_key = str(idx)
                cached = self.evidence_cache.get(cache_key)
                if cached:
                    evidence_feat = torch.tensor(cached['evidence_feat'], dtype=torch.float32)
                    if evidence_feat.dim() == 2:
                        evidence_feat = evidence_feat.squeeze(0)
                    out['evidence_feat'] = evidence_feat.cpu()
                    logger.debug(f"从缓存加载证据特征: idx={idx}")
                    return out
            
            proof = row.get('proof', '')
            logger.debug(f"处理证据特征, 是否使用LLM={self.llm_wrapper is not None}")
            
            evidence_feat, llm_retrieval_result = proof_to_evidence_feat(
                proof, 
                self.evidence_dim, 
                self.llm_wrapper,
                news_text=text,
            )
            
            out['evidence_feat'] = evidence_feat.float().cpu()
            logger.debug(f"证据特征处理完成: LLM结果={llm_retrieval_result is not None}")
        else:
            out['evidence_feat'] = torch.zeros(self.evidence_dim).cpu()
        return out


def load_train_dataset():
    import logging
    logger = logging.getLogger(__name__)
    
    train_csv_path = None
    
    weibo_path = os.path.join(PROJECT_ROOT, 'processed_weibo', 'weibo_train.csv')
    tiktok_path = os.path.join(PROJECT_ROOT, 'processed_tiktok', 'tiktok_train.csv')
    
    if os.path.exists(weibo_path):
        train_csv_path = weibo_path
        logger.info(f"使用微博训练数据集: {weibo_path}")
    elif os.path.exists(tiktok_path):
        train_csv_path = tiktok_path
        logger.info(f"使用TikTok训练数据集: {tiktok_path}")
    else:
        logger.warning(f"未找到训练数据集文件，检查路径: {weibo_path}, {tiktok_path}")
        return []
    
    samples = []
    try:
        with open(train_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append({
                    'text': row.get('text', ''),
                    'image': row.get('image', ''),
                    'video': row.get('video', ''),
                    'label': int(row.get('label', 0))
                })
        
        logger.info(f"成功加载训练数据集，共 {len(samples)} 条样本")
        return samples
        
    except Exception as e:
        logger.error(f"加载训练数据集失败: {e}")
        import traceback
        traceback.print_exc()
        return []
