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


########################data process########################
from torch.utils.data import DataLoader
import torch.optim as optim

def input_data_process(features, labels):
    """
    输入数据处理函数
    把正常的数据集转化为torch.utils.data.Dataset类型
    输入的数据集格式必须为：torch.Tensor类型；numpy.ndarray类型；list类型
    输入结果为：torch.utils.data.Dataset类型
    """
    class MyDataSet(torch.utils.data.Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels
        def __len__(self):
            return len(self.features)
        def __getitem__(self, i):
            return self.features[i], self.labels[i]
        
    #把输入数据转化为torch.Tensor类型
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    #求features中所有数据的最小值和最大值
    min_vals, _ = torch.min(features, dim=0)  # dim=0表示在列方向上计算
    max_vals, _ = torch.max(features, dim=0)
    #对features进行min-max归一化处理
    features = (features - min_vals) / (max_vals - min_vals)
    #对features在列方向上进行标准化处理
    features = (features - features.mean(dim=0)) / features.std(dim=0)

    
    dataset = MyDataSet(features, labels)

    #获取features的维度
    input_size = features.shape[1]
    #获取labels的类别数量
    num_classes = len(labels.unique())

    return dataset, input_size, num_classes


def softmax_regression(features_train, labels_train, features_test, labels_test, hidden_size1, hidden_size2, num_epochs, momentum, batch_size, learning_rate):
    """
    多层感知机（MLP）分类模型
    输入变量包括：
    features_train:训练集特征数据
    labels_train:训练集标签数据
    features_test:测试集特征数据
    labels_test:测试集标签数据
    hidden_size1:隐藏层1的神经元数量
    hidden_size2:隐藏层2的神经元数量
    num_epochs:训练轮次
    momentum:动量参数
    batch_size:批次大小
    learning_rate:学习率

    输出结果为：训练集和测试集的准确率
    """ 
    train_dataset, input_size, num_classes = input_data_process(features_train, labels_train)
    test_dataset, _, _ = input_data_process(features_test, labels_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
            super(NeuralNet, self).__init__()  # 调用父类的初始化方法
            self.flatten = nn.Flatten()  
            self.layer1 = nn.Linear(input_size, hidden_size1)  # 第一个全连接层
            self.relu1 = nn.ReLU()  # 第一个ReLU激活函数
            self.layer2 = nn.Linear(hidden_size1, hidden_size2)  # 第二个全连接层
            self.relu2 = nn.ReLU()  # 第二个ReLU激活函数
            self.output = nn.Linear(hidden_size2, num_classes)  # 输出层

        def forward(self, x):
            x = self.flatten(x)         # 展平输入图像
            x = self.relu1(self.layer1(x))  # 第一层：线性变换后接ReLU
            x = self.relu2(self.layer2(x))  # 第二层：线性变换后接ReLU
            x = self.output(x)          # 输出层：线性变换
            return x
        
    model = NeuralNet(input_size, hidden_size1, hidden_size2, num_classes)

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数（内部包含softmax）
    optimizer = optim.SGD(
        model.parameters(),    # 模型参数
        lr=learning_rate,      # 学习率
        momentum=momentum      # 动量参数
    )

    def train_model():
        model.train()  # 设置模型为训练模式
        total_step = len(train_loader)  # 计算每个epoch的总批次数
        
        for epoch in range(num_epochs):  # 遍历每个训练轮次
            for i, (features, labels) in enumerate(train_loader):  # 遍历每个批次
                # 前向传播
                outputs = model(features)  # 通过模型获取预测结果
                loss = criterion(outputs, labels)  # 计算损失
                
                # 反向传播和优化
                optimizer.zero_grad()  # 清空之前的梯度
                loss.backward()        # 反向传播计算梯度
                optimizer.step()       # 更新模型参数
                
                # 每100个批次打印一次训练信息
                if (i+1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], '
                        f'Step [{i+1}/{total_step}], '
                        f'Loss: {loss.item():.4f}')

    def test_model():
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 关闭梯度计算
            correct = 0  # 正确预测的样本数
            total = 0    # 总样本数
            
            for features, labels in test_loader:  # 遍历测试集
                outputs = model(features)  # 获取模型预测结果
                _, predicted = torch.max(outputs.data, 1)  # 获取最大概率的类别
                #outputs.data是输出结果，每一行对应一个样本在每一个类别上的概率
                #dim=1表示在行维度上取最大的值 
                #torch.max函数返回两个值，第一个值是最大值，第二个值是最大值的索引
                #predicted是最大概率的类别索引，所以最终的结果是返回一个包括所有最大值索引的向量
                total += labels.size(0)  # 累加样本数
                correct += (predicted == labels).sum().item()  # 累加正确预测数
        
        # 打印总体准确率
        print(f'总体准确率: {100 * correct / total:.2f}%')


    print("开始训练...")
    train_model()  # 训练模型
    print("\n开始测试...")
    test_model()   # 测试模型

