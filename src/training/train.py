# -*- coding: utf-8 -*-
"""主训练脚本模块"""

import os
import sys
import random
import numpy as np
import torch

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from transformers import BertTokenizer
from torch.utils.data import DataLoader

from src.data.dataset import MultimodalFakeNewsDataset, get_train_transform
from src.models.feature_extractors import get_bert_extractor, get_resnet50_extractor
from src.models.multimodal_fusion_model import MultimodalFusionModel, MultimodalFusionWithExtractors
from src.training.config import get_train_config, get_model_config
from src.training.trainer import Trainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(train_config, model_config, device):
    bert_path = train_config.get('bert_path') or os.path.join(PROJECT_ROOT, 'models', 'bert-base-chinese')
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    text_encoder = get_bert_extractor(bert_path, device)  
    resnet = get_resnet50_extractor(pretrained=True, device=device)
    video_encoder = resnet
    
    fusion = MultimodalFusionModel(
        text_dim=128,
        visual_dim=2048,
        evidence_dim=256,
        text_expert_dim=128,
        visual_expert_dim=256,
        evidence_expert_dim=256,
        dropout=model_config.get('dropout', 0.3),
    )
    
    llm_wrapper = None
    try:
        import transformers_stream_generator
        logger.info("transformers_stream_generator已安装")
        
        from src.llm.qwen_wrapper import QwenWrapper
        llm_model_path = train_config.get('llm_path') or os.path.join(PROJECT_ROOT, 'models', 'Qwen1.5-4B-Chat')
        logger.info(f"尝试加载Qwen模型: {llm_model_path}")
        
        if not os.path.exists(llm_model_path):
            logger.error(f"Qwen模型目录不存在: {llm_model_path}")
        else:
            files = os.listdir(llm_model_path)
            logger.info(f"Qwen模型目录文件: {files[:10]}...")
            llm_wrapper = QwenWrapper(model_path=llm_model_path, device=device)
            logger.info("Qwen模型加载成功！")
    except ImportError as e:
        logger.error(f"缺少依赖项: {e}")
    except Exception as e:
        logger.error(f"无法加载Qwen模型: {e}")
        import traceback
        traceback.print_exc()
    
    model = MultimodalFusionWithExtractors(
        text_encoder=text_encoder,
        image_encoder=resnet,
        fusion_model=fusion,
        video_encoder=video_encoder,
        llm_wrapper=llm_wrapper,
    )
    return model, tokenizer


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="多模态虚假新闻检测模型训练")
    parser.add_argument('--dataset', type=str, default='weibo', choices=['weibo', 'tiktok'],
                      help='选择数据集: weibo 或 tiktok')
    parser.add_argument('--resume', type=str, default=None,
                      help='从检查点恢复训练，指定检查点路径')
    parser.add_argument('--resume-latest', action='store_true',
                      help='从最新的检查点恢复训练')
    parser.add_argument('--use-cache', action='store_true',
                      help='使用预计算的证据特征缓存')
    parser.add_argument('--cache-dir', type=str, default=None,
                      help='证据特征缓存目录')
    args = parser.parse_args()
    
    train_config = get_train_config()
    model_config = get_model_config()
    
    set_seed(train_config.get('seed', 42))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.warning(f"使用设备: {device}")
    logger.warning(f"选择的数据集: {args.dataset}")
    
    try:
        model, tokenizer = build_model(train_config, model_config, device)
        logger.warning("模型构建成功")
    except Exception as e:
        logger.error(f"模型构建失败: {e}")
        return
    
    if args.dataset == 'weibo':
        data_root = train_config.get('data_root_weibo') or os.path.join(PROJECT_ROOT, 'processed_weibo')
        train_csv = os.path.join(data_root, 'weibo_train.csv')
        val_csv = os.path.join(data_root, 'weibo_val.csv')
        process_script = 'process_weibo.py'
    else:
        data_root = train_config.get('data_root_tiktok') or os.path.join(PROJECT_ROOT, 'processed_tiktok')
        train_csv = os.path.join(data_root, 'tiktok_train.csv')
        val_csv = os.path.join(data_root, 'tiktok_val.csv')
        process_script = 'process_tiktok.py'
    
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        logger.error(f"数据文件不存在: {train_csv} 或 {val_csv}")
        logger.info(f"请先运行 {process_script} 生成处理后的数据")
        return
    
    try:
        max_text_len = train_config.get('max_text_len', 512)
        
        cache_dir = args.cache_dir or os.path.join(data_root, 'evidence_cache')
        train_cache = os.path.join(cache_dir, f'{args.dataset}_train_evidence_cache.json') if args.use_cache else None
        val_cache = os.path.join(cache_dir, f'{args.dataset}_val_evidence_cache.json') if args.use_cache else None
        
        if args.use_cache:
            logger.warning(f"使用缓存模式，缓存目录: {cache_dir}")
            if train_cache is None or val_cache is None:
                logger.error("缓存路径未正确设置，请检查 --cache-dir 参数或配置文件")
                return
            if not os.path.exists(train_cache) or not os.path.exists(val_cache):
                logger.error(f"缓存文件不存在！请先运行: python src/training/precompute_evidence.py --dataset {args.dataset}")
                logger.error(f"缺少: {train_cache} 或 {val_cache}")
                return
        
        llm_wrapper = model.llm_wrapper if hasattr(model, 'llm_wrapper') else None
        if not args.use_cache:
            logger.info(f"LLM包装器状态: {'已加载' if llm_wrapper is not None else '未加载'}")
        else:
            logger.info("使用缓存模式，训练时不需要LLM推理")
        
        train_ds = MultimodalFakeNewsDataset(
            train_csv, data_root, tokenizer,
            max_text_len=max_text_len,
            image_transform=get_train_transform(),
            llm_wrapper=None if args.use_cache else llm_wrapper,
            evidence_cache_path=train_cache
        )
        val_ds = MultimodalFakeNewsDataset(
            val_csv, data_root, tokenizer, 
            max_text_len=max_text_len, 
            llm_wrapper=None if args.use_cache else llm_wrapper,
            evidence_cache_path=val_cache
        )
        
        batch_size = train_config.get('batch_size', 16)
        num_workers = train_config.get('num_workers', 4)
        pin_memory = train_config.get('pin_memory', True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        
        logger.warning(f"数据集加载成功: 训练集 {len(train_ds)} 样本, 验证集 {len(val_ds)} 样本")
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        return
    
    try:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_config,
            device=device,
            checkpoint_dir=train_config.get('checkpoint_dir'),
        )
        logger.warning("训练器创建成功")
    except Exception as e:
        logger.error(f"训练器创建失败: {e}")
        return
    
    try:
        logger.warning("开始训练...")
        epochs = train_config.get('epochs', 50)
        
        resume_from = None
        if args.resume_latest:
            latest_path = os.path.join(train_config.get('checkpoint_dir', 'checkpoints'), 'latest_checkpoint.pt')
            if os.path.exists(latest_path):
                resume_from = latest_path
                logger.warning(f"将从最新检查点恢复: {latest_path}")
            else:
                logger.warning("未找到最新检查点，将从头开始训练")
        elif args.resume:
            resume_from = args.resume
            logger.warning(f"将从指定检查点恢复: {resume_from}")
        
        trainer.train(epochs=epochs, resume_from=resume_from)
        logger.warning("训练完成！")
    except Exception as e:
        logger.error(f"训练失败: {e}")
        return
    
    logger.info("\n=== 训练完成 ===")
    logger.info(f"模型已训练 {epochs} 轮")
    logger.info("训练过程已完成，模型已保存")


if __name__ == '__main__':
    main()
