import argparse
import torch
import torch.nn as nn
from pathlib import Path
import logging
from typing import Dict, Any
import optuna
from datetime import datetime
import itertools
import copy

from data import DataModule
from model import ModelBuilder
from train import Trainer
from evaluate import Evaluator
from utils import Utils
from config import ConfigManager

logger = logging.getLogger(__name__)

def grid_search(config: Dict, data_module: DataModule) -> Dict:
    """
    执行网格搜索

    Args:
        config: 配置字典
        data_module: 数据模块

    Returns:
        包含所有组合及其验证集性能的字典
    """
    search_space = config['optimization']['grid_search']
    keys = list(search_space.keys())
    values = list(search_space.values())
    combinations = list(itertools.product(*values))

    results = {}
    for combo in combinations:
        params = dict(zip(keys, combo))
        logger.info(f"Testing combination: {params}")

        # 使用深拷贝避免修改原始配置
        trial_config = copy.deepcopy(config)
        trial_config['training'].update(params)

        # 构建模型
        model_builder = ModelBuilder(trial_config)
        model = model_builder.build_model()

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

        # 训练模型
        trainer = Trainer(model, data_module.train_loader, data_module.val_loader,
                         criterion, optimizer, trial_config)
        trainer.train()

        # 记录结果，使用元组作为键
        params_tuple = tuple(sorted(params.items()))
        results[params_tuple] = trainer.best_val_acc

    return results

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
    # 使用深拷贝避免修改原始配置
    trial_config = copy.deepcopy(config)

    # 从配置文件中读取贝叶斯优化的搜索空间
    search_space = trial_config['optimization']['bayesian']['search_space']

    params = {}
    for param, details in search_space.items():
        if details['type'] == 'float':
            try:
                low = float(details['low'])
                high = float(details['high'])
            except ValueError as ve:
                logger.error(f"Parameter {param} has invalid float values: low={details['low']}, high={details['high']}")
                raise ve
            log = details.get('log', False)
            # 添加调试日志
            logger.debug(f"Suggesting float for {param}: low={low}, high={high}, log={log}")
            params[param] = trial.suggest_float(param, low, high, log=log)
        elif details['type'] == 'categorical':
            params[param] = trial.suggest_categorical(param, details['choices'])
        else:
            raise ValueError(f"Unsupported parameter type: {details['type']}")

    # 更新配置
    trial_config['training'].update(params)

    # 构建模型
    model_builder = ModelBuilder(trial_config)
    model = model_builder.build_model()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    # 训练模型
    trainer = Trainer(model, data_module.train_loader, data_module.val_loader,
                     criterion, optimizer, trial_config)
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
    config_dict = config.dict()

    # 调试日志：打印 Bayesian search_space 的内容和类型
    bayesian_search_space = config_dict['optimization']['bayesian']['search_space']
    for param, details in bayesian_search_space.items():
        if details['type'] == 'float':
            low = details['low']
            high = details['high']
            logger.debug(f"Param: {param}, low: {low} ({type(low)}), high: {high} ({type(high)})")

    # 创建实验目录
    exp_dir = Utils.create_experiment_dir()
    config_dict['save_dir'] = str(exp_dir)

    # 准备数据
    data_module = DataModule(config_dict)
    try:
        data_module.setup()
    except Exception as e:
        logger.error(f"Error setting up data module: {str(e)}")
        raise

    if mode == 'train' or mode == 'optimize':
            # 训练或优化模式不需要预训练模型
            model_builder = ModelBuilder(config_dict)
            model = model_builder.build_model()

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['training']['learning_rate'])

            if mode == 'train':
                trainer = Trainer(model, data_module.train_loader, data_module.val_loader,
                                criterion, optimizer, config_dict)
                trainer.train()

                # 评估
                evaluator = Evaluator(model, data_module.test_loader, criterion, config_dict)
                results = evaluator.evaluate()
                Utils.save_results(results, exp_dir / 'final_results.json')

            elif mode == 'optimize':
                # 超参数优化模式
                search_method = config.optimization.search_method  # 选择优化方法：'bayesian' 或 'grid'
                if search_method == 'bayesian':
                    # 贝叶斯优化模式
                    study = Utils.setup_optuna_study(
                        study_name=f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        direction='maximize'
                    )
                    study.optimize(
                        lambda trial: objective(trial, config_dict, data_module),
                        n_trials=config.optimization.bayesian.n_trials,
                        timeout=config.optimization.bayesian.timeout
                    )
                    # 保存优化结果
                    optimization_results = {
                        'best_params': study.best_params,
                        'best_value': study.best_value,
                        'best_trial': study.best_trial.number
                    }
                    Utils.save_results(optimization_results, exp_dir / 'optimization_results.json')

                    # 使用最佳参数训练最终模型
                    best_params = study.best_params
                    trial_config = copy.deepcopy(config_dict)
                    trial_config['training'].update(best_params)
                    model_builder = ModelBuilder(trial_config)
                    model = model_builder.build_model()

                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])

                    trainer = Trainer(model, data_module.train_loader, data_module.val_loader,
                                    criterion, optimizer, trial_config)
                    trainer.train()

                    # 评估最终模型
                    evaluator = Evaluator(model, data_module.test_loader, criterion, trial_config)
                    results = evaluator.evaluate()
                    Utils.save_results(results, exp_dir / 'final_model_results.json')

                elif search_method == 'grid':
                    # 网格搜索模式
                    grid_results = grid_search(config_dict, data_module)
                    Utils.save_results(grid_results, exp_dir / 'grid_search_results.json')

                    # 找到最佳参数
                    best_combo = max(grid_results, key=grid_results.get)
                    best_params = dict(best_combo)  # 将元组转换为字典
                    logger.info(f"Best parameters from grid search: {best_params}")

                    # 使用最佳参数训练最终模型
                    trial_config = copy.deepcopy(config_dict)
                    trial_config['training'].update(best_params)
                    model_builder = ModelBuilder(trial_config)
                    model = model_builder.build_model()

                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])

                    trainer = Trainer(model, data_module.train_loader, data_module.val_loader,
                                    criterion, optimizer, trial_config)
                    trainer.train()

                    # 评估最终模型
                    evaluator = Evaluator(model, data_module.test_loader, criterion, trial_config)
                    results = evaluator.evaluate()
                    Utils.save_results(results, exp_dir / 'final_model_results.json')

                else:
                    raise ValueError(f"Unsupported search method: {search_method}")

    elif mode == 'evaluate':
        # 评估模式
        model_builder = ModelBuilder(config_dict)
        model = model_builder.build_model()

        # 加载预训练模型
        Utils.load_model(model, config_dict['model']['pretrained_path'])

        criterion = nn.CrossEntropyLoss()
        evaluator = Evaluator(model, data_module.test_loader, criterion, config_dict)

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