import yaml
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any
from typing import Optional

class BayesianConfig(BaseModel):
    n_trials: int
    timeout: int
    search_space: Dict[str, Dict[str, Any]]

class GridSearchConfig(BaseModel):
    learning_rate: List[float]
    batch_size: List[int]
    hidden_size: List[int]
    dropout_rate: List[float]

class OptimizationConfig(BaseModel):
    search_method: Literal['bayesian', 'grid']
    bayesian: BayesianConfig = Field(default=None)
    grid_search: GridSearchConfig = Field(default=None)

class DataConfig(BaseModel):
    data_path: str
    label_column: str
    train_ratio: float
    val_ratio: float
    batch_size: int
    normalize: bool
    num_workers: int
    pin_memory: bool

class ModelConfig(BaseModel):
    type: str
    input_size: int
    hidden_size: int
    num_classes: int
    dropout_rate: float
    pretrained_path: Optional[str] = None  # 可选字段

class TrainingConfig(BaseModel):
    learning_rate: float
    num_epochs: int
    device: str
    early_stopping_patience: int
    checkpoint_interval: int
    plot_interval: int

class EvaluationConfig(BaseModel):
    bootstrap_iterations: int
    cv_folds: int

class Config(BaseModel):
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    optimization: OptimizationConfig
    save_dir: str

class ConfigManager:
    @staticmethod
    def load_config(config_path: str) -> Config:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return Config(**config_dict)