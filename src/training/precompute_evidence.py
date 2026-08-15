# -*- coding: utf-8 -*-
"""预计算证据特征脚本"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import json
import argparse
import torch
import csv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import BertTokenizer
from src.llm.qwen_wrapper import QwenWrapper
from src.llm.evidence_extractor import proof_to_evidence_feat
from src.training.config import get_train_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def process_single_sample(args):
    # 处理单个样本，返回证据特征
    idx, row, llm_wrapper, evidence_dim, device = args
    
    text = row.get('text', '')
    proof = row.get('proof', '')
    label = int(row.get('label', 0))
    
    try:
        evidence_feat, llm_result = proof_to_evidence_feat(
            proof=proof,
            evidence_dim=evidence_dim,
            llm_wrapper=llm_wrapper,
            news_text=text,
            image_caption='',
            ocr_text='',
            video_summary=''
        )
        
        return {
            'idx': idx,
            'evidence_feat': evidence_feat.cpu().numpy().tolist(),
            'success': True
        }
    except Exception as e:
        logger.error(f"样本 {idx} 处理失败: {e}")
        return {
            'idx': idx,
            'evidence_feat': [[0.0] * evidence_dim],
            'success': False
        }


def precompute_evidence_features(
    csv_path: str,
    output_path: str,
    llm_wrapper,
    evidence_dim: int = 256,
    batch_size: int = 1,
):
    # 预计算所有样本的证据特征
    samples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        samples = list(reader)
    
    logger.info(f"加载了 {len(samples)} 个样本")
    
    device = llm_wrapper.device if llm_wrapper else torch.device('cpu')
    cache = {}
    
    for idx, row in enumerate(tqdm(samples, desc="预计算证据特征")):
        result = process_single_sample((idx, row, llm_wrapper, evidence_dim, device))
        cache[str(idx)] = {
            'evidence_feat': result['evidence_feat'],
        }
        
        if (idx + 1) % 100 == 0:
            logger.info(f"已处理 {idx + 1}/{len(samples)} 个样本")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    
    logger.info(f"缓存已保存到: {output_path}")
    
    return cache


def main():
    parser = argparse.ArgumentParser(description='预计算证据特征')
    parser.add_argument('--dataset', type=str, default='weibo', choices=['weibo', 'tiktok'],
                        help='数据集类型')
    parser.add_argument('--split', type=str, default='all', 
                        choices=['train', 'val', 'test', 'all'],
                        help='数据集划分')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--evidence-dim', type=int, default=256,
                        help='证据特征维度')
    args = parser.parse_args()
    
    config = get_train_config()
    
    if args.dataset == 'weibo':
        data_root = config.get('data_root_weibo') or os.path.join(PROJECT_ROOT, 'processed_weibo')
        csv_files = {
            'train': os.path.join(data_root, 'weibo_train.csv'),
            'val': os.path.join(data_root, 'weibo_val.csv'),
            'test': os.path.join(data_root, 'weibo_test.csv'),
        }
    else:
        data_root = config.get('data_root_tiktok') or os.path.join(PROJECT_ROOT, 'processed_tiktok')
        csv_files = {
            'train': os.path.join(data_root, 'tiktok_train.csv'),
            'val': os.path.join(data_root, 'tiktok_val.csv'),
            'test': os.path.join(data_root, 'tiktok_test.csv'),
        }
    
    output_dir = args.output_dir or os.path.join(data_root, 'evidence_cache')
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    llm_model_path = config.get('llm_path') or os.path.join(PROJECT_ROOT, 'models', 'Qwen1.5-4B-Chat')
    logger.info(f"加载LLM模型: {llm_model_path}")
    
    llm_wrapper = QwenWrapper(model_path=llm_model_path, device=device)
    
    splits_to_process = ['train', 'val', 'test'] if args.split == 'all' else [args.split]
    
    for split in splits_to_process:
        csv_path = csv_files.get(split)
        if not csv_path or not os.path.exists(csv_path):
            logger.warning(f"数据集文件不存在: {csv_path}")
            continue
        
        output_path = os.path.join(output_dir, f'{args.dataset}_{split}_evidence_cache.json')
        
        logger.info(f"\n{'='*50}")
        logger.info(f"处理 {args.dataset} {split} 集")
        logger.info(f"输入: {csv_path}")
        logger.info(f"输出: {output_path}")
        logger.info(f"{'='*50}\n")
        
        precompute_evidence_features(
            csv_path=csv_path,
            output_path=output_path,
            llm_wrapper=llm_wrapper,
            evidence_dim=args.evidence_dim,
        )
    
    logger.info("\n所有数据集处理完成！")


if __name__ == '__main__':
    main()
