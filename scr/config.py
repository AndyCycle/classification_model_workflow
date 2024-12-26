import yaml
from dataclasses import asdict, dataclass, field
from typing import Dict, Any

@dataclass
class DataConfig:
    data_path: str
    label_column: str
    train_ratio: float
    val_ratio: float
    batch_size: int
    normalize: bool
    num_workers: int
    pin_memory: bool

@dataclass
class ModelConfig:
    type: str
    input_size: int
    hidden_size: int
    num_classes: int
    dropout_rate: float
    pretrained_path: str = None

@dataclass
class TrainingConfig:
    learning_rate: float
    num_epochs: int
    device: str
    early_stopping_patience: int
    checkpoint_interval: int
    plot_interval: int

@dataclass
class EvaluationConfig:
    bootstrap_iterations: int
    cv_folds: int

@dataclass
class OptimizationConfig:
    n_trials: int
    timeout: int

@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    optimization: OptimizationConfig
    save_dir: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ConfigManager:
    @staticmethod
    def load_config(config_path: str) -> Config:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return Config(
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            evaluation=EvaluationConfig(**config_dict['evaluation']),
            optimization=OptimizationConfig(**config_dict['optimization']),
            save_dir=config_dict['save_dir']
        )

