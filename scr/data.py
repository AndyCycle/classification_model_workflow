import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from typing import Tuple, Optional, Union, List
import os

logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    """
    自定义数据集类，支持多种数据格式的加载和预处理

    Attributes:
        data: 特征数据
        labels: 标签数据
        transform: 数据转换函数
        feature_scaler: 特征标准化器
    """

    def __init__(self,
                 data_path: str,
                 label_column: str = 'label',
                 transform: Optional[callable] = None,
                 normalize: bool = True) -> None:
        """
        初始化数据集

        Args:
            data_path: 数据文件路径
            label_column: 标签列名
            transform: 数据转换函数
            normalize: 是否进行标准化
        """
        self.transform = transform
        self.feature_scaler = StandardScaler() if normalize else None

        # 加载数据
        self.data, self.labels = self._load_data(data_path, label_column)
        logger.info(f"Dataset loaded with {len(self.data)} samples")

    def _load_data(self,
                   data_path: str,
                   label_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载数据文件

        Args:
            data_path: 数据文件路径
            label_column: 标签列名

        Returns:
            特征数据和标签数据的元组
        """
        file_extension = os.path.splitext(data_path)[1]

        try:
            if file_extension == '.csv':
                df = pd.read_csv(data_path)
            elif file_extension == '.npy':
                df = pd.DataFrame(np.load(data_path))
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # 分离特征和标签
            labels = df[label_column].values
            features = df.drop(columns=[label_column]).values

            # 标签编码
            self.label_encoder = LabelEncoder()
            labels = self.label_encoder.fit_transform(labels)  # M=1, B=0

            # 标准化特征（如果需要）
            if self.feature_scaler is not None:
                features = self.feature_scaler.fit_transform(features)

            return features.astype(np.float32), labels.astype(np.int64)

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个数据样本

        Args:
            idx: 样本索引

        Returns:
            特征和标签的元组
        """
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return torch.FloatTensor(sample), torch.tensor(label, dtype=torch.long)

    def get_feature_dim(self) -> int:
        """返回特征维度"""
        return self.data.shape[1]

class DataModule:
    """
    数据模块类，负责数据集的划分和加载器的创建
    """

    def __init__(self, config: dict) -> None:
        """
        初始化数据模块

        Args:
            config: 配置字典，包含数据相关的所有配置
        """
        self.config = config
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def setup(self) -> None:
        """
        设置数据集和数据加载器
        """
        try:
            # 创建数据集
            self.dataset = CustomDataset(
                data_path=self.config['data']['data_path'],
                label_column=self.config['data']['label_column'],
                normalize=self.config['data']['normalize']
            )

            # 计算数据集划分大小
            total_size = len(self.dataset)
            train_size = int(self.config['data']['train_ratio'] * total_size)
            val_size = int(self.config['data']['val_ratio'] * total_size)
            test_size = total_size - train_size - val_size

            # 划分数据集
            train_dataset, val_dataset, test_dataset = random_split(
                self.dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)  # 设置随机种子以确保可重复性
            )

            # 创建数据加载器
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['data']['batch_size'],
                shuffle=True,
                num_workers=self.config['data']['num_workers'],
                pin_memory=self.config['data']['pin_memory']
            )

            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['data']['batch_size'],
                shuffle=False,
                num_workers=self.config['data']['num_workers'],
                pin_memory=self.config['data']['pin_memory']
            )

            self.test_loader = DataLoader(
                test_dataset,
                batch_size=self.config['data']['batch_size'],
                shuffle=False,
                num_workers=self.config['data']['num_workers'],
                pin_memory=self.config['data']['pin_memory']
            )

            logger.info(f"Data split completed - Train: {train_size}, Val: {val_size}, Test: {test_size}")

        except Exception as e:
            logger.error(f"Error setting up data module: {str(e)}")
            raise

    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        获取数据加载器

        Returns:
            训练、验证和测试数据加载器的元组
        """
        if not all([self.train_loader, self.val_loader, self.test_loader]):
            raise RuntimeError("Data loaders not initialized. Call setup() first.")
        return self.train_loader, self.val_loader, self.test_loader

    def get_feature_dim(self) -> int:
        """
        获取特征维度

        Returns:
            特征维度
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")
        return self.dataset.get_feature_dim()

# 示例配置文件中的数据相关配置
example_config = {
    'data': {
        'data_path': 'data/breast_cancer.csv',
        'label_column': 'diagnosis',
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'batch_size': 32,
        'normalize': True,
        'num_workers': 4,
        'pin_memory': True
    }
}

def get_data_loaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    便捷函数，用于获取数据加载器

    Args:
        config: 配置字典

    Returns:
        训练、验证和测试数据加载器的元组
    """
    data_module = DataModule(config)
    data_module.setup()
    return data_module.get_loaders()
