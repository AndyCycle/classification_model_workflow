import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import KFold
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional
import logging
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd

logger = logging.getLogger(__name__)

class Evaluator:
    """
    模型评估器类，负责模型评估、ROC曲线绘制、bootstrapping分析和交叉验证
    """

    def __init__(self,
                 model: torch.nn.Module,
                 test_loader: DataLoader,
                 criterion: torch.nn.Module,
                 config: dict) -> None:
        """
        初始化评估器

        Args:
            model: 待评估的模型
            test_loader: 测试数据加载器
            criterion: 损失函数
            config: 评估配置
        """
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)

        # 创建保存目录
        self.save_dir = Path(config.get('save_dir', 'evaluation_results'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Evaluator initialized. Using device: {self.device}")

    def evaluate(self) -> Dict:
        """
        在测试集上评估模型

        Returns:
            包含评估指标的字典
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc="Evaluating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        # 计算评估指标
        results = self._calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )

        # 绘制ROC曲线
        self._plot_roc_curve(results['fpr'], results['tpr'], results['auc'])

        # 绘制混淆矩阵
        self._plot_confusion_matrix(results['confusion_matrix'])

        # 保存结果
        self._save_results(results)

        return results

    def bootstrap_evaluation(self, n_iterations: int = 1000) -> Dict:
        """
        使用bootstrapping方法评估模型的稳定性

        Args:
            n_iterations: bootstrapping迭代次数

        Returns:
            包含bootstrapping结果的字典
        """
        logger.info(f"Starting bootstrapping analysis with {n_iterations} iterations")

        # 收集所有数据
        all_data = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                all_data.append(inputs)
                all_labels.append(labels)

        all_data = torch.cat(all_data)
        all_labels = torch.cat(all_labels)

        # Bootstrapping
        metrics = {
            'auc': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

        for i in tqdm(range(n_iterations), desc="Bootstrapping"):
            # 随机采样
            indices = np.random.choice(len(all_labels), len(all_labels), replace=True)
            batch_data = all_data[indices].to(self.device)
            batch_labels = all_labels[indices].to(self.device)

            # 预测
            outputs = self.model(batch_data)
            probs = torch.softmax(outputs, dim=1)
            probs = probs.detach()
            _, predicted = torch.max(outputs, 1)

            # 计算指标
            batch_results = self._calculate_metrics(
                batch_labels.cpu().numpy(),
                predicted.cpu().numpy(),
                probs[:, 1].cpu().numpy()
            )

            for metric in metrics:
                if metric in batch_results:
                    metrics[metric].append(batch_results[metric])

        # 计算统计量
        bootstrap_results = {
            metric: {
                'median': np.median(values),
                'std': np.std(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5)
            }
            for metric, values in metrics.items()
        }

        # 绘制分布图
        self._plot_bootstrap_distributions(metrics)

        # 保存结果
        self._save_bootstrap_results(bootstrap_results)

        return bootstrap_results

    def cross_validation(self,
                        dataset: torch.utils.data.Dataset,
                        n_splits: int = 5,
                        batch_size: int = 32) -> Dict:
        """
        执行交叉验证

        Args:
            dataset: 完整数据集
            n_splits: 折数
            batch_size: 批次大小

        Returns:
            包含交叉验证结果的字典
        """
        logger.info(f"Starting {n_splits}-fold cross validation")

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_results = {
            'fold_metrics': [],
            'fold_predictions': [],
            'fold_labels': [],
            'fold_probabilities': []
        }

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            logger.info(f"Processing fold {fold + 1}/{n_splits}")

            # 创建数据加载器
            val_sampler = SubsetRandomSampler(val_idx)
            val_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=val_sampler
            )

            # 评估当前折
            fold_preds = []
            fold_labels = []
            fold_probs = []

            self.model.eval()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)

                    fold_preds.extend(predicted.cpu().numpy())
                    fold_labels.extend(labels.cpu().numpy())
                    fold_probs.extend(probs[:, 1].cpu().numpy())

            # 计算当前折的指标
            fold_metrics = self._calculate_metrics(
                np.array(fold_labels),
                np.array(fold_preds),
                np.array(fold_probs)
            )

            cv_results['fold_metrics'].append(fold_metrics)
            cv_results['fold_predictions'].append(fold_preds)
            cv_results['fold_labels'].append(fold_labels)
            cv_results['fold_probabilities'].append(fold_probs)

            logger.info(f"Fold {fold + 1} AUC: {fold_metrics['auc']:.4f}")

        # 计算平均指标
        mean_metrics = self._calculate_mean_metrics(cv_results['fold_metrics'])

        # 绘制所有折的ROC曲线
        self._plot_cv_roc_curves(cv_results)

        # 计算预测值与真实标签的相关系数
        all_preds = np.concatenate(cv_results['fold_probabilities'])
        all_labels = np.concatenate(cv_results['fold_labels'])
        correlation, p_value = stats.pearsonr(all_preds, all_labels)

        cv_results['mean_metrics'] = mean_metrics
        cv_results['correlation'] = correlation
        cv_results['p_value'] = p_value

        # 保存结果
        self._save_cv_results(cv_results)

        return cv_results

    def _calculate_metrics(self,
                         labels: np.ndarray,
                         predictions: np.ndarray,
                         probabilities: np.ndarray) -> Dict:
        """
        计算评估指标

        Args:
            labels: 真实标签
            predictions: 预测标签
            probabilities: 预测概率

        Returns:
            包含各种指标的字典
        """
        fpr, tpr, _ = roc_curve(labels, probabilities)
        confusion = confusion_matrix(labels, predictions)
        classification_rep = classification_report(labels, predictions, output_dict=True)

        return {
            'accuracy': classification_rep['accuracy'],
            'precision': classification_rep['weighted avg']['precision'],
            'recall': classification_rep['weighted avg']['recall'],
            'f1': classification_rep['weighted avg']['f1-score'],
            'auc': auc(fpr, tpr),
            'fpr': fpr,
            'tpr': tpr,
            'confusion_matrix': confusion
        }

    def _plot_roc_curve(self,
                       fpr: np.ndarray,
                       tpr: np.ndarray,
                       auc_score: float,
                       title: str = "ROC Curve") -> None:
        """绘制ROC曲线"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        plt.close()

    def _plot_confusion_matrix(self, cm: np.ndarray) -> None:
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.save_dir / 'confusion_matrix.png')
        plt.close()

    def _plot_bootstrap_distributions(self, metrics: Dict) -> None:
        """绘制bootstrapping分布图"""
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))

        for ax, (metric_name, values) in zip(axes, metrics.items()):
            sns.histplot(values, ax=ax, kde=True)
            ax.axvline(np.median(values), color='r', linestyle='--')
            ax.set_title(f'{metric_name.capitalize()} Distribution')
            ax.set_xlabel(metric_name)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'bootstrap_distributions.png')
        plt.close()

    def _plot_cv_roc_curves(self, cv_results: Dict) -> None:
        """绘制交叉验证ROC曲线"""
        plt.figure(figsize=(8, 6))

        # 绘制每一折的ROC曲线
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(mean_fpr)

        for fold, metrics in enumerate(cv_results['fold_metrics']):
            # Interpolate tpr values
            interp_tpr = np.interp(mean_fpr, metrics['fpr'], metrics['tpr'])
            interp_tpr[0] = 0.0  # Ensure the curve starts at (0,0)
            mean_tpr += interp_tpr

            plt.plot(
                metrics['fpr'],
                metrics['tpr'],
                alpha=0.3,
                label=f'ROC fold {fold+1} (AUC = {metrics["auc"]:.4f})'
            )

        # 绘制平均ROC曲线
        mean_tpr /= len(cv_results['fold_metrics'])
        mean_tpr[-1] = 1.0  # Ensure the curve ends at (1,1)

        mean_auc = np.trapz(mean_tpr, mean_fpr)
        plt.plot(
            mean_fpr,
            mean_tpr,
            color='b',
            label=f'Mean ROC (AUC = {mean_auc:.4f})',
            linewidth=2,
            alpha=0.8
        )

        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Cross-Validation ROC Curves')
        plt.legend(loc="lower right")
        plt.savefig(self.save_dir / 'cv_roc_curves.png')
        plt.close()

    def _calculate_mean_metrics(self, fold_metrics: List[Dict]) -> Dict:
        """计算交叉验证的平均指标"""
        mean_metrics = {}
        for metric in fold_metrics[0].keys():
            if metric not in ['fpr', 'tpr', 'confusion_matrix']:
                values = [fold[metric] for fold in fold_metrics]
                mean_metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        return mean_metrics

    def _save_results(self, results: Dict) -> None:
        """保存评估结果"""
        results_copy = results.copy()
        # 转换numpy数组为列表以便JSON序列化
        for key in results_copy:
            if isinstance(results_copy[key], np.ndarray):
                results_copy[key] = results_copy[key].tolist()

        with open(self.save_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results_copy, f, indent=4)

    def _save_bootstrap_results(self, results: Dict) -> None:
        """保存bootstrapping结果"""
        with open(self.save_dir / 'bootstrap_results.json', 'w') as f:
            json.dump(results, f, indent=4)

    def _save_cv_results(self, results: Dict) -> None:
        """保存交叉验证结果"""
        # 创建可序列化的结果副本
        serializable_results = {
            'mean_metrics': results['mean_metrics'],
            'correlation': float(results['correlation']),
            'p_value': float(results['p_value'])
        }

        with open(self.save_dir / 'cv_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=4)
