﻿import argparse
import torch
import torch.nn as nn
from pathlib import Path
import logging
from typing import Dict, Any
import optuna
from datetime import datetime

from data import DataModule
from model import ModelBuilder
from train import Trainer
from evaluate import Evaluator
from utils import Utils
from config import ConfigManager

logger = logging.getLogger(__name__)

def objective(trial: optuna.Trial, config: Dict, data_module: DataModule) -> float:
    """
    Optuna优化目标函数

    Args:
        trial: Optuna试验对象
        config: 配置字典
        data_module: 数据模块

    Returns:
        验证集上的性能指标
    """
    # 定义超参数搜索空间
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256, 512]),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5)
    }

    # 更新配置
    config['training'].update(params)

    # 构建模型
    model_builder = ModelBuilder(config)
    model = model_builder.build_model()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    # 训练模型
    trainer = Trainer(model, data_module.train_loader, data_module.val_loader,
                     criterion, optimizer, config)
    trainer.train()

    return trainer.best_val_acc

def main(config_path: str, mode: str) -> None:
    """
    主函数

    Args:
        config_path: 配置文件路径
        mode: 运行模式 ('train', 'evaluate', 'optimize')
    """
    # 设置统一日志
    Utils.setup_logging()

    # 加载配置
    config = ConfigManager.load_config(config_path)

    # 创建实验目录
    exp_dir = Utils.create_experiment_dir()
    config['save_dir'] = str(exp_dir)

    # 准备数据
    data_module = DataModule(config)
    data_module.setup()

    if mode == 'train':
        # 训练模式
        model_builder = ModelBuilder(config)
        model = model_builder.build_model()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

        trainer = Trainer(model, data_module.train_loader, data_module.val_loader,
                         criterion, optimizer, config)
        trainer.train()

        # 评估
        evaluator = Evaluator(model, data_module.test_loader, criterion, config)
        results = evaluator.evaluate()

        # 保存结果
        Utils.save_results(results, exp_dir / 'final_results.json')

    elif mode == 'evaluate':
        # 评估模式
        model_builder = ModelBuilder(config)
        model = model_builder.build_model()

        # 加载预训练模型
        Utils.load_model(model, config['model']['pretrained_path'])

        criterion = nn.CrossEntropyLoss()
        evaluator = Evaluator(model, data_module.test_loader, criterion, config)

        # 进行完整评估
        results = evaluator.evaluate()
        bootstrap_results = evaluator.bootstrap_evaluation()
        cv_results = evaluator.cross_validation(data_module.dataset)

        # 保存所有结果
        all_results = {
            'standard_evaluation': results,
            'bootstrap_results': bootstrap_results,
            'cross_validation': cv_results
        }
        Utils.save_results(all_results, exp_dir / 'evaluation_results.json')

    elif mode == 'optimize':
        # 超参数优化模式
        study = Utils.setup_optuna_study(
            study_name=f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            direction='maximize'
        )

        # 运行优化
        study.optimize(
            lambda trial: objective(trial, config, data_module),
            n_trials=config['optimization']['n_trials']
        )

        # 保存优化结果
        optimization_results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial': study.best_trial.number
        }
        Utils.save_results(optimization_results, exp_dir / 'optimization_results.json')

        # 使用最佳参数训练最终模型
        config['training'].update(study.best_params)
        model_builder = ModelBuilder(config)
        model = model_builder.build_model()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=study.best_params['learning_rate'])

        trainer = Trainer(model, data_module.train_loader, data_module.val_loader,
                         criterion, optimizer, config)
        trainer.train()

        # 评估最终模型
        evaluator = Evaluator(model, data_module.test_loader, criterion, config)
        results = evaluator.evaluate()
        Utils.save_results(results, exp_dir / 'final_model_results.json')

    else:
        raise ValueError(f"Unknown mode: {mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Learning Binary Classification')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'optimize'],
                        default='train', help='Running mode')
    args = parser.parse_args()

    try:
        main(args.config, args.mode)
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise
