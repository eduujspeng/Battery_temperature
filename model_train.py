# -*- coding:utf-8 -*-
# author:peng
# Date：2023/12/11 16:07
import torch
from tqdm import tqdm

# 定义训练函数
from tqdm import tqdm


def model_train(model, optimizer, scheduler, train_loader, num_epochs, epoch, criterion, device):
    # 使用tqdm创建进度条
    pbar = tqdm(train_loader)

    # 训练模型
    model.train()

    total_loss = 0
    index = 0
    # 遍历训练数据加载器
    for batch_X, batch_y in pbar:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 模型前向传播
        outputs = model(batch_X)

        # 计算损失
        loss = criterion(outputs, batch_y)
        total_loss += loss.item()
        index += 1

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 更新进度条显示
        pbar.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
        pbar.set_postfix(**{'total_loss': total_loss / (index + 1),
                            'lr': optimizer.param_groups[0]['lr']})

    # 学习率调度器更新
    scheduler.step()


# 定义测试函数
from tqdm import tqdm

def model_test(model, test_loader, criterion, device):
    # 使用tqdm创建进度条
    pbar = tqdm(test_loader)
    print('test start!!!!')

    # 模型评估
    model.eval()
    total_loss = 0
    index = 0

    # 遍历测试数据加载器
    for batch_X, batch_y in pbar:
        with torch.no_grad():
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # 模型前向传播
            outputs = model(batch_X)

            # 计算损失
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            index += 1

        # 更新进度条显示
        pbar.set_postfix(**{'total_loss': total_loss / (index + 1)})





