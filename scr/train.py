import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
import os
from datetime import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Trainer:
    """
    模型训练器类，负责模型训练、验证和检查点保存
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: Optimizer,
                 config: dict,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> None:
        """
        初始化训练器

        Args:
            model: 待训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            config: 训练配置
            scheduler: 学习率调度器（可选）
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        # 设置设备
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)

        # 训练状态跟踪
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

        # 创建保存目录
        self.save_dir = Path(config.get('save_dir', 'checkpoints'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Trainer initialized. Using device: {self.device}")

    def train(self) -> None:
        """
        执行完整的训练过程
        """
        num_epochs = self.config['training']['num_epochs']
        early_stopping_patience = self.config['training']['early_stopping_patience']
        no_improvement_count = 0

        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # 训练一个epoch
            train_loss, train_acc = self._train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # 验证
            val_loss, val_acc = self._validate_epoch()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # 学习率调整
            if self.scheduler is not None:
                self.scheduler.step()

            # 保存检查点
            improved = self._save_checkpoint(val_loss, val_acc)

            # 早停检查
            if not improved:
                no_improvement_count += 1
                if no_improvement_count >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            else:
                no_improvement_count = 0

            # 绘制训练进度图
            if (epoch + 1) % self.config.get('plot_interval', 5) == 0:
                self._plot_training_progress()

        # 训练结束，保存最终结果
        self._save_training_results()
        logger.info("Training completed")

    def _train_epoch(self) -> Tuple[float, float]:
        """
        训练一个epoch

        Returns:
            epoch的平均损失和准确率
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # 使用tqdm创建进度条
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新进度条
            current_loss = total_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        logger.info(f'Train Epoch: {self.current_epoch + 1} '
                   f'Loss: {epoch_loss:.4f} '
                   f'Acc: {epoch_acc:.2f}%')

        return epoch_loss, epoch_acc

    def _validate_epoch(self) -> Tuple[float, float]:
        """
        验证一个epoch

        Returns:
            验证集的平均损失和准确率
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_loss = total_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        logger.info(f'Validation Epoch: {self.current_epoch + 1} '
                   f'Loss: {epoch_loss:.4f} '
                   f'Acc: {epoch_acc:.2f}%')

        return epoch_loss, epoch_acc

    def _save_checkpoint(self, val_loss: float, val_acc: float) -> bool:
        """
        保存检查点

        Args:
            val_loss: 验证损失
            val_acc: 验证准确率

        Returns:
            是否是最佳模型
        """
        improved = False

        # 检查是否是最佳模型
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            improved = True

            # 保存最佳模型
            checkpoint = {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': self.config
            }

            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

            torch.save(checkpoint, self.save_dir / 'best_model.pth')
            logger.info(f'Saved best model with validation accuracy: {val_acc:.2f}%')

        # 定期保存检查点
        if (self.current_epoch + 1) % self.config.get('checkpoint_interval', 10) == 0:
            checkpoint = {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': self.config
            }

            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

            torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{self.current_epoch + 1}.pth')

        return improved

    def _plot_training_progress(self) -> None:
        """
        绘制训练进度图
        """
        plt.figure(figsize=(12, 4))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc')
        plt.plot(self.val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Training and Validation Accuracy')

        plt.tight_layout()
        plt.savefig(self.save_dir / f'training_progress_epoch_{self.current_epoch + 1}.png')
        plt.close()

    def _save_training_results(self) -> None:
        """
        保存训练结果
        """
        results = {
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc,
            'total_epochs': self.current_epoch + 1,
            'config': self.config
        }

        # 保存为JSON文件
        with open(self.save_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=4)

        logger.info(f"Training results saved to {self.save_dir / 'training_results.json'}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载检查点

        Args:
            checkpoint_path: 检查点文件路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['val_acc']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
