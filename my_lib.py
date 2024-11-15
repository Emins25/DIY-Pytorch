import numpy as np
import torch
from torch.utils import data
from torch import nn

#######################linear regression#######################

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays) 
    #data.TensorDataset将张量组合成数据集
    #*data_arrays解包数据数组元组
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
    #data.DataLoader创建一个数据加载器，用于批量处理数据
    #batch_size指定每个批次的数据量
    #shuffle=is_train表示是否在每个epoch重新打乱数据

def linreg(features, labels, num_epochs, batch_size, learning_rate):
    """
    线性回归函数用于通过梯度下降方法求解线性回归问题
    输入变量包括：
    features:特征数据集
    labels:标签数据集
    num_epochs:训练轮次
    batch_size:批次大小
    learning_rate:学习率
    """
    #data process
    data_iter = load_array((features, labels), batch_size)
    net = nn.Sequential(nn.Linear(len(features[0]), 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')
