# -*- coding: utf-8 -*-
"""训练器模块"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from .config import get_train_config
from ..utils.metrics import compute_metrics
from ..utils.logger import get_logger
import matplotlib.pyplot as plt
import json

from torch.cuda.amp import autocast, GradScaler

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config=None,
        device=None,
        checkpoint_dir=None,
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or get_train_config()
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = self.model.to(self.device)
        self.checkpoint_dir = checkpoint_dir or self.config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 2e-5),
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(0.9, 0.999),
        )
        self.scheduler = None
        if self.config.get('scheduler') == 'ReduceLROnPlateau':
            sk = self.config.get('scheduler_kwargs', {})
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=sk.get('mode', 'min'),
                factor=sk.get('factor', 0.5),
                patience=sk.get('patience', 3),
            )
        self.grad_clip = self.config.get('grad_clip', 1.0)
        self.early_stop_patience = self.config.get('early_stopping_patience', 5)
        self.early_stop_metric = self.config.get('early_stopping_metric', 'f1')
        self.best_metric = -1.0
        self.patience_counter = 0 
        self.use_amp = self.config.get('use_amp', False) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        self.history = {
            'train_loss': [],
            'train_f1': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_f1': [],
            'val_accuracy': [],
            'val_auc': []
        }
        
        from datetime import datetime
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.result_dir = os.path.join(self.config.get('result_dir', 'results'), self.timestamp)
        os.makedirs(self.result_dir, exist_ok=True)
        
        tensorboard_dir = os.path.join(self.config.get('tensorboard_dir', 'runs'), self.timestamp)
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        all_preds = torch.tensor([], device=self.device)
        all_labels = torch.tensor([], device=self.device)
        n_batches = 0
        total_batches = len(self.train_loader)
        
        logger.info(f"开始训练第 {epoch} 轮，共 {total_batches} 个批次")
        
        for i, batch in enumerate(self.train_loader, 1):
            if i % 10 == 0 or i == total_batches:
                logger.info(f"正在处理训练批次 {i}/{total_batches}")
            
            self.optimizer.zero_grad()
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            image = batch['image']
            labels = batch['label']
            has_image = batch.get('has_image', torch.zeros_like(labels, device=self.device))
            has_video = batch.get('has_video', torch.zeros_like(labels, device=self.device))
            evidence_feat = batch.get('evidence_feat', torch.zeros(input_ids.size(0), 256, device=self.device))
            
            if self.use_amp and self.scaler is not None:
                with autocast():
                    final_score, weights = self.model(
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
                    loss = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                final_score, weights = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image=image,
                    evidence_feat=evidence_feat,
                    video_path=batch.get('video_path'),
                    text=batch.get('text_raw'),
                    has_image=has_image,
                    has_video=has_video,
                )
                logits = torch.cat([1 - final_score, final_score], dim=1)  # [real_prob, fake_prob]
                loss = self.criterion(logits, labels)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            
            all_preds = torch.cat([all_preds, preds], dim=0)
            all_labels = torch.cat([all_labels, labels], dim=0)
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        metrics = compute_metrics(all_preds, all_labels)
        metrics['loss'] = avg_loss
        
        self.history['train_loss'].append(metrics['loss'])
        self.history['train_f1'].append(metrics['f1'])
        self.history['train_accuracy'].append(metrics['accuracy'])

        current_lr = self.optimizer.param_groups[0]['lr']
        logger.warning(f"第 {epoch} 轮训练完成，损失: {metrics['loss']:.4f}, F1: {metrics['f1']:.4f}, Acc:{metrics['accuracy']:.4f}, LR: {current_lr:.2e}")
        return metrics

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        all_preds = torch.tensor([], device=self.device)
        all_labels = torch.tensor([], device=self.device)
        all_scores = torch.tensor([], device=self.device)
        total_loss = 0.0
        n_batches = 0
        total_batches = len(self.val_loader)
        
        logger.info(f"开始验证，共 {total_batches} 个批次")
        
        for i, batch in enumerate(self.val_loader, 1):
            if i % 10 == 0 or i == total_batches:
                logger.info(f"正在处理验证批次 {i}/{total_batches}")
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            image = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            has_image = batch.get('has_image', torch.zeros_like(labels, device=self.device))
            has_video = batch.get('has_video', torch.zeros_like(labels, device=self.device))
            evidence_feat = batch.get('evidence_feat', torch.zeros(batch['input_ids'].size(0), 256, device=self.device))
            
            if self.use_amp:
                with autocast():
                    final_score, weights = self.model(
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
                    loss = self.criterion(logits, labels)
            else:
                final_score, weights = self.model(
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
                loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            scores = final_score.squeeze(-1)
            
            all_preds = torch.cat([all_preds, preds], dim=0)
            all_labels = torch.cat([all_labels, labels], dim=0)
            all_scores = torch.cat([all_scores, scores], dim=0)
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        metrics = compute_metrics(all_preds, all_labels, scores=all_scores)
        metrics['loss'] = avg_loss
        
        self.history['val_loss'].append(metrics['loss'])
        self.history['val_f1'].append(metrics['f1'])
        self.history['val_accuracy'].append(metrics['accuracy'])
        self.history['val_auc'].append(metrics.get('auc', 0.0))
        
        logger.warning(f"验证完成，损失: {metrics['loss']:.4f}, F1: {metrics['f1']:.4f}, 准确率: {metrics['accuracy']:.4f}, AUC: {metrics.get('auc', 0):.4f}")
        return metrics

    def save_checkpoint(self, path, epoch, extra=None):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_f1': self.best_metric,
        }
        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        if extra:
            state.update(extra)
        torch.save(state, path)

    def load_checkpoint(self, path):
        return torch.load(path, map_location=self.device)
    
    def resume_from_checkpoint(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in state and self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
        
        if 'history' in state:
            self.history = state['history']
        if 'best_f1' in state:
            self.best_metric = state['best_f1']
        if 'patience_counter' in state:
            self.patience_counter = state['patience_counter']
        
        start_epoch = state.get('epoch', 0) + 1
        logger.warning(f"从检查点恢复: epoch {state.get('epoch', 0)}, best_f1={self.best_metric:.4f}")
        return start_epoch
    
    def save_visualizations(self):
        config_path = os.path.join(self.result_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        
        metrics_path = os.path.join(self.result_dir, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        
        plt.figure(figsize=(12, 8))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Loss vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.history['train_f1'], 'b-', label='Training F1')
        plt.plot(epochs, self.history['val_f1'], 'r-', label='Validation F1')
        plt.title('F1 Score vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.history['train_accuracy'], 'b-', label='Train Accuracy')
        plt.plot(epochs, self.history['val_accuracy'], 'g-', label='Val Accuracy')
        plt.title('Accuracy vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        curves_path = os.path.join(self.result_dir, 'training_curves.png')
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        metrics_table = []
        for i, (train_loss, train_f1, val_loss, val_f1, val_acc) in enumerate(zip(
            self.history['train_loss'],
            self.history['train_f1'],
            self.history['val_loss'],
            self.history['val_f1'],
            self.history['val_accuracy']
        )):
            metrics_table.append({
                'epoch': i+1,
                'train_loss': train_loss,
                'train_f1': train_f1,
                'val_loss': val_loss,
                'val_f1': val_f1,
                'val_accuracy': val_acc
            })
        
        csv_path = os.path.join(self.result_dir, 'metrics.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('Epoch,Train Loss,Train F1,Train Acc,Val Loss,Val F1,Val Accuracy\n')
            for entry in metrics_table:
                f.write(f"{entry['epoch']},{entry['train_loss']:.4f},{entry['train_f1']:.4f},{entry['train_acc']:.4f},{entry['val_loss']:.4f},{entry['val_f1']:.4f},{entry['val_accuracy']:.4f}\n")
        
        logger.info(f"可视化结果已保存到: {self.result_dir}")

    def train(self, epochs=None, resume_from=None):
        epochs = epochs or self.config.get('epochs', 30)
        start_epoch = 1
        
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.resume_from_checkpoint(resume_from)
            logger.warning(f"从 epoch {start_epoch} 继续训练")
        
        for epoch in range(start_epoch, epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['loss'])
            
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('F1/train', train_metrics['f1'], epoch)
            self.writer.add_scalar('F1/val', val_metrics['f1'], epoch)
            self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

            logger.warning(
                f"Epoch {epoch} | train loss={train_metrics['loss']:.4f} f1={train_metrics['f1']:.4f} | "
                f"val loss={val_metrics['loss']:.4f} f1={val_metrics['f1']:.4f} acc={val_metrics['accuracy']:.4f} auc={val_metrics.get('auc', 0):.4f}"
            )
            current = val_metrics.get(self.early_stop_metric, val_metrics['f1'])
            if current > self.best_metric:
                self.best_metric = current
                self.patience_counter = 0
                best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                self.save_checkpoint(best_path, epoch, extra={'config': self.config, 'val_metrics': val_metrics})
                logger.warning(f"  -> 保存最佳模型 (best {self.early_stop_metric}={current:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stop_patience:
                    logger.warning(f"早停于 epoch {epoch}")
                    break
            if epoch % 5 == 0:
                ckpt_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
                self.save_checkpoint(ckpt_path, epoch)
        
        self.writer.close()
        self.save_visualizations()
        return self.best_metric
