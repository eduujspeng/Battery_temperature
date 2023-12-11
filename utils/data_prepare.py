# -*- coding:utf-8 -*-
# author:peng
# Date：2023/12/11 16:09
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def read_csv(path):
    # 读取CSV文件
    df = pd.read_csv(path, header=None)

    # 为数据添加表头
    header = ['绝对时间', '流程相对时间', '步次相对时间', '步次序号', '步次类型', '电压', '电流', '线电压', '容量',
              '能量', '功率', '通道温度', '电池温度', '接触阻抗']
    df.columns = header

    # 选择影响电池温度合适的相关因素
    df_new = df[['电压', '电流', '线电压', '容量',
                 '能量', '功率', '通道温度', '接触阻抗', '电池温度']]
    return df_new


# 构建输入序列和目标序列，X为过去的10个时间步的特征，包含电池温度，y为第11个时间步的电池温度特征
def make_sequence(data, time_step=10):
    X, y = [], []

    # 遍历数据集，构建输入和目标序列
    for i in range(len(data) - time_step):
        _X = data[i: (i + time_step), :]
        X.append(_X)
        y.append(data[i + time_step, -1])

    # 数据进行预处理归一化，避免某些因素数值较大而对结果产生较大的比重
    scaler_X = MinMaxScaler()
    X = scaler_X.fit_transform(np.array(X).reshape(-1, 1)).reshape(-1, time_step, 9)
    y = np.array(y).reshape(-1, 1)

    return X, y
