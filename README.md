
# 二分类预测模型

## 项目简介
本项目利用深度学习模型构建分类预测系统，支持灵活的模型搭建、训练、测试及评估。

## 项目架构
- `config.py`：配置文件
- `data.py`：数据处理与加载
- `model.py`：模型构建与管理
- `train.py`：模型训练
- `evaluate.py`：模型评估
- `utils.py`：工具函数
- `main.py`：主控脚本

## 功能说明
### config.py
- **Config**：配置文件

### data.py
- **DataLoader**：处理数据集划分和加载。

### model.py
- **ModelBuilder**：构建和管理不同的深度学习模型。

### train.py
- **Trainer**：负责模型的训练过程，输出每个epoch的loss和accuracy。

### evaluate.py
- **Evaluator**：评估模型性能，包括ROC曲线、AUC值、Bootstrapping和交叉验证。

### utils.py
- **Utils**：保存最佳模型、绘制图表等工具功能。

### main.py
- 实现模型替换和超参数搜索（网格搜索、贝叶斯搜索）。

## 使用方法
1. 配置`config.yaml`中的参数。
2. 用`main.py --config your/path/to/config.yaml`开始训练和评估。
3. 用`main.py --mode optimize`进行超参数优化。
4. 用`main.py --mode train`进行模型测试。
5. 用`main.py --mode evaluate`进行模型测试。