import torch

from model.GRU import Model
from matplotlib import pyplot as plt

from train import get_loader

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型、损失函数和优化器参数
    input_size = 9  # 输入特征的数量
    hidden_size = 50  # 隐藏层的大小
    output_size = 1  # 输出的大小（电池温度）
    num_layers = 2  # RNN的层数

    # 定义model
    model = Model(input_size, hidden_size, output_size, num_layers).to(device)

    # 加载权重
    model.load_state_dict(torch.load('weight/Battery_last.pth', map_location=device))

    # 加载数据集
    train_loader, test_loader = get_loader()

    pre_ls = []
    label_ls = []
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            outputs_np = outputs.squeeze().detach().cpu().numpy()

            pre_ls.extend(outputs_np)

            label_np = batch_y.squeeze().cpu().numpy()

            label_ls.extend(label_np)

    print(len(pre_ls), len(label_ls))
    print('predict:',pre_ls[:3], 'label:', label_ls[:3])

    plt.figure(figsize=(12, 4))
    plt.title('Battery Temperature Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Battery Temperature')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(label_ls[:10], color='red', label='Actual')
    plt.plot(pre_ls[:10], color='green', label='predicted')
    plt.legend()
    plt.show()
