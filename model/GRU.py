# -*- coding:utf-8 -*-
# author:peng
# Date：2023/12/11 15:59
# 定义GRU模型
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Model, self).__init__()

        # 定义GRU模型，其中：
        # - input_size：输入特征的维度
        # - hidden_size：GRU隐藏层的维度
        # - num_layers：GRU的层数
        # - batch_first=True 表示输入的数据格式为 (batch_size, seq_length, input_size)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        # 定义全连接层，将GRU的输出映射到最终的输出维度
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 前向传播过程
        # x 是输入序列数据，维度为 (batch_size, seq_length, input_size)

        # 将输入序列数据传入GRU模型
        out, _ = self.rnn(x)

        # 仅保留每个样本序列中最后一个时间步的输出
        # 在 batch_first=True 的情况下，out 的维度为 (batch_size, seq_length, hidden_size)
        # 所以选择 out[:, -1, :] 表示每个样本序列的最后一个时间步的输出
        out = self.fc(out[:, -1, :])

        # 返回最终的输出
        return out
