# 电池温度预测项目

## 项目简介

本项目使用GRU（Gated Recurrent Unit）神经网络来预测电池温度。其中输入X为过去的10个时间步的特征，包含电池温度，y为第11个时间步的电池温度特征。
GRU是一种循环神经网络（RNN）的变体，适用于序列数据的建模和预测。

## 文件结构

- `data/`: 存放数据集的文件夹
- `model/`: 存放训练好的模型的文件夹
    - `model.py`: GRU模型的定义
- `utils/`: 存放一些工具函数的文件夹
    - `data_preprocessing.py`: 数据预处理脚本
- `weight/`: 存放一些权重的文件夹
- `model_train.py`: 模型训练和测试的函数定义
- `train.py`: 模型训练和测试的脚本
- `predict.py`: 使用训练好的模型进行预测的脚本
- `README.md`: 项目说明文档

## 使用说明

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 模型训练：
```bash
python train.py
```

3. 模型预测：
```
python predict.py
```

## 注意事项
- 请根据实际情况调整模型参数和训练配置。
- 数据集应包含时间序列特征和电池温度作为目标。