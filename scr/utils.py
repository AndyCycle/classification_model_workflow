import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import optuna
from sklearn.model_selection import ParameterGrid
import pandas as pd
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class Utils:
    """工具类，提供各种通用功能"""

    @staticmethod
    def setup_logging(log_file: str = 'project.log') -> None:
        """
        设置统一的日志记录器

        Args:
            log_file: 日志文件路径
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger()
        logger.info("Logging is configured.")

    @staticmethod
    def load_config(config_path: str) -> Dict:
        """
        加载配置文件————改为使用config.py中的ConfigManager

        Args:
            config_path: 配置文件路径

        Returns:
            配置字典
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    @staticmethod
    def save_model(model: torch.nn.Module,
                  path: str,
                  optimizer: Optional[torch.optim.Optimizer] = None,
                  additional_info: Optional[Dict] = None) -> None:
        """
        保存模型和相关信息

        Args:
            model: PyTorch模型
            path: 保存路径
            optimizer: 优化器（可选）
            additional_info: 其他需要保存的信息（可选）
        """
        save_dict = {
            'model_state_dict': model.state_dict(),
            'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        }

        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()

        if additional_info is not None:
            save_dict.update(additional_info)

        try:
            torch.save(save_dict, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    @staticmethod
    def load_model(model: torch.nn.Module,
                  path: str,
                  optimizer: Optional[torch.optim.Optimizer] = None) -> Optional[Dict]:
        """
        加载模型和相关信息

        Args:
            model: PyTorch模型
            path: 模型文件路径
            optimizer: 优化器（可选）

        Returns:
            包含加载信息的字典或None
        """
        if not path:
            logger.info("No pretrained model path provided, using initialized model.")
            return None

        if not os.path.isfile(path):
            logger.error(f"Model checkpoint file does not exist at path: {path}")
            return None

        try:
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])

            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            logger.info(f"Model loaded from {path}")
            return checkpoint
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    @staticmethod
    def plot_roc_curves(curves_data: List[Dict],
                       save_path: str,
                       title: str = "ROC Curves Comparison") -> None:
        """
        绘制多个ROC曲线进行比较

        Args:
            curves_data: ROC曲线数据列表
            save_path: 图片保存路径
            title: 图表标题
        """
        plt.figure(figsize=(10, 8))

        for data in curves_data:
            plt.plot(data['fpr'], data['tpr'],
                    label=f"{data['name']} (AUC = {data['auc']:.4f})")

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")

        try:
            plt.savefig(save_path)
            plt.close()
            logger.info(f"ROC curves plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving ROC curves plot: {str(e)}")
            raise

    @staticmethod
    def plot_training_history(history: Dict,
                            save_path: str) -> None:
        """
        绘制训练历史

        Args:
            history: 训练历史数据
            save_path: 图片保存路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 损失曲线
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # 准确率曲线
        ax2.plot(history['train_acc'], label='Train Accuracy')
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        try:
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Training history plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving training history plot: {str(e)}")
            raise

    @staticmethod
    def grid_search(param_grid: Dict) -> List[Dict]:
        """
        生成网格搜索参数组合

        Args:
            param_grid: 参数网格字典

        Returns:
            参数组合列表
        """
        return list(ParameterGrid(param_grid))

    @staticmethod
    def create_experiment_dir(base_dir: str = 'experiments') -> Path:
        """
        创建实验目录

        Args:
            base_dir: 基础目录路径

        Returns:
            实验目录路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = Path(base_dir) / timestamp
        exp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created experiment directory: {exp_dir}")
        return exp_dir

    @staticmethod
    def save_results(results: Dict, save_path: str) -> None:
        """
        保存结果到JSON文件

        Args:
            results: 结果字典
            save_path: 保存路径
        """
        try:
            results_copy = Utils._convert_ndarray(results)
            with open(save_path, 'w') as f:
                json.dump(results_copy, f, indent=4, default=Utils._json_default)
            logger.info(f"Results saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

    @staticmethod
    def _convert_ndarray(obj):
        """
        递归将字典中的numpy.ndarray转换为列表

        Args:
            obj: 需要转换的对象

        Returns:
            转换后的对象
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: Utils._convert_ndarray(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Utils._convert_ndarray(item) for item in obj]
        return obj

    @staticmethod
    def _json_default(obj):
        """
        JSON序列化的默认处理函数

        Args:
            obj: 需要序列化的对象

        Returns:
            可序列化的对象
        """
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    @staticmethod
    def setup_optuna_study(study_name: str,
                          direction: str = 'maximize') -> optuna.Study:
        """
        设置Optuna研究

        Args:
            study_name: 研究名称
            direction: 优化方向 ('maximize' 或 'minimize')

        Returns:
            Optuna研究对象
        """
        try:
            study = optuna.create_study(
                study_name=study_name,
                direction=direction,
                storage=f"sqlite:///{study_name}.db",
                load_if_exists=True
            )
            logger.info(f"Created/loaded Optuna study: {study_name}")
            return study
        except Exception as e:
            logger.error(f"Error setting up Optuna study: {str(e)}")
            raise
