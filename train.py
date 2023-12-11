# -*- coding:utf-8 -*-
# author:peng
# Date：2023/12/11 17:01
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch import optim, nn
from model.GRU import Model
from model_train import model_train, model_test
from utils.data_prepare import read_csv, make_sequence

def get_loader():
    # 加载数据集
    path1 = 'data/1.csv'
    path2 = 'data/4.csv'
    df1, df4 = read_csv(path1), read_csv(path2)

    # 将数据转为numpy并合并为一个
    data1 = df1.values
    data4 = df4.values
    data = np.concatenate((data1, data4), axis=0)

    # 构建输入序列和目标序列，X为过去的10个时间步的特征，包含电池温度，y为第11个时间步的电池温度特征
    X, y = make_sequence(data)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 将numpy转换为PyTorch的Tensor
    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.Tensor(y_train)
    X_test_tensor = torch.Tensor(X_test)
    y_test_tensor = torch.Tensor(y_test)

    # 将数据转换为DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_loader, test_loader

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型、损失函数和优化器参数
    input_size = 9  # 输入特征的数量
    hidden_size = 50  # 隐藏层的大小
    output_size = 1  # 输出的大小（电池温度）
    num_layers = 2  # RNN的层数
    learning_rate = 1e-3

    # 定义模型、损失函数和优化器
    model = Model(input_size, hidden_size, output_size, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9, last_epoch=-1)
    criterion = nn.MSELoss().to(device)

    # 加载数据集
    train_loader, test_loader = get_loader()

    # 模型训练与测试
    num_epochs = 50

    for epoch in range(num_epochs):
        model_train(model, optimizer, scheduler, train_loader, num_epochs, epoch, criterion, device)

        if (epoch + 1) % 10 == 0:
            model_test(model, test_loader, criterion, device)
    torch.save(model.state_dict(), 'Battery_last.pth')

