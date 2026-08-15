# -*- coding: utf-8 -*-
"""在测试集上评估最佳模型"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from transformers import BertTokenizer
from src.data.dataset import MultimodalFakeNewsDataset
from src.models.feature_extractors import get_bert_extractor, get_resnet50_extractor
from src.models.multimodal_fusion_model import MultimodalFusionModel, MultimodalFusionWithExtractors
from src.training.config import get_train_config, get_model_config
from src.training.checkpoint import load_checkpoint
from src.utils.metrics import compute_metrics, classification_report_str
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_model_and_data(config_train, config_model, checkpoint_path, device, use_cache=False, cache_dir=None):
    bert_path = config_train.get('bert_path') or os.path.join(PROJECT_ROOT, 'models', 'bert-base-chinese')
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
        dropout=config_model.get('dropout', 0.3),
    )
    
    from src.llm.qwen_wrapper import QwenWrapper
    llm_model_path = config_train.get('llm_path') or os.path.join(PROJECT_ROOT, 'models', 'Qwen1.5-4B-Chat')
    llm_wrapper = QwenWrapper(model_path=llm_model_path, device=device)
    
    model = MultimodalFusionWithExtractors(
        text_encoder=text_encoder,
        image_encoder=resnet,
        fusion_model=fusion,
        video_encoder=video_encoder,
        llm_wrapper=llm_wrapper,
    )
    state, _ = load_checkpoint(checkpoint_path, model=model, device=device)
    model = model.to(device)
    model.eval()
    data_root = config_train.get('data_root_weibo') or os.path.join(PROJECT_ROOT, 'processed_weibo')
    test_csv = os.path.join(data_root, 'weibo_test.csv')
    if not os.path.exists(test_csv):
        test_csv = os.path.join(PROJECT_ROOT, 'processed_weibo', 'weibo_test.csv')
        data_root = os.path.join(PROJECT_ROOT, 'processed_weibo')
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"测试集不存在: {test_csv}，请先运行 process_weibo.py 生成测试集")
    
    test_cache = None
    if use_cache:
        cache_dir = cache_dir or os.path.join(data_root, 'evidence_cache')
        test_cache = os.path.join(cache_dir, 'weibo_test_evidence_cache.json')
        if not os.path.exists(test_cache):
            logger.warning(f"测试集缓存不存在: {test_cache}，将使用实时计算")
            test_cache = None
    
    dataset = MultimodalFakeNewsDataset(
        test_csv, data_root, tokenizer,
        max_text_len=config_train.get('max_text_len', 128),
        llm_wrapper=None if use_cache and test_cache else llm_wrapper,
        evidence_cache_path=test_cache
    )
    loader = DataLoader(dataset, batch_size=config_train.get('batch_size', 16), shuffle=False)
    return model, loader, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--output', type=str, default=None, help='结果输出目录')
    parser.add_argument('--use-cache', action='store_true', help='使用预计算的证据特征缓存')
    parser.add_argument('--cache-dir', type=str, default=None, help='证据特征缓存目录')
    args = parser.parse_args()
    config_train = get_train_config()
    config_model = get_model_config()
    checkpoint_path = args.checkpoint or os.path.join(config_train.get('checkpoint_dir'), 'best_model.pt')
    if not os.path.exists(checkpoint_path):
        logger.error("检查点不存在: %s", checkpoint_path)
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model, loader, _ = load_model_and_data(
            config_train, config_model, checkpoint_path, device,
            use_cache=args.use_cache, cache_dir=args.cache_dir
        )
    except FileNotFoundError as e:
        logger.error("%s", e)
        return
    all_preds = []
    all_labels = []
    all_scores = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            image = batch['image']
            labels = batch['label']
            has_image = batch.get('has_image')
            has_video = batch.get('has_video')

            evidence_feat = batch.get('evidence_feat', torch.zeros(batch['input_ids'].size(0), 256, device=device))
            
            final_score, weights = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image=image,
                evidence_feat=evidence_feat,
                video_path=batch.get('video_path'),
                text=batch.get('text_raw'),
                has_image=has_image,
                has_video=has_video,
            )
            logits = torch.cat([1 - final_score, final_score], dim=1)  
            preds = logits.argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(labels)
            all_scores.append(final_score.squeeze(-1))
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    metrics = compute_metrics(all_preds, all_labels, scores=all_scores)
    report = classification_report_str(all_preds, all_labels)
    logger.info("===== 测试集评估结果 =====")
    logger.info("Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | AUC: %.4f",
                metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1'], metrics.get('auc', 0))
    logger.info("\n%s", report)
    out_dir = args.output or config_train.get('result_dir')
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'eval_metrics.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1: {metrics['f1']:.4f}\n")
            f.write(f"AUC: {metrics.get('auc', 0):.4f}\n\n")

            f.write(report)
        logger.info("结果已保存到 %s", out_dir)


if __name__ == '__main__':
    main()
