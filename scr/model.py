import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class ModelBuilder:
    """
    模型构建器类，负责根据配置文件构建不同的模型
    """

    def __init__(self, config: dict):
        """
        初始化模型构建器

        Args:
            config: 配置字典
        """
        self.config = config

    def build_model(self) -> nn.Module:
        """
        构建模型

        Returns:
            构建好的PyTorch模型
        """
        model_type = self.config['model']['type']

        if model_type == 'MLP':
            input_dim = self.config['model']['input_size']
            hidden_size = self.config['model']['hidden_size']
            num_classes = self.config['model']['num_classes']
            dropout_rate = self.config['model']['dropout_rate']

            model = MLP(input_dim, hidden_size, num_classes, dropout_rate)
            logger.info("MLP model built successfully")
            return model
        else:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")

class MLP(nn.Module):
    """
    简单的多层感知机模型
    """
    def __init__(self, input_dim: int, hidden_size: int, num_classes: int, dropout_rate: float = 0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out