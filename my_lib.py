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
    输出结果为：torch.utils.data.Dataset类型
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

#####################################################################################################################################






#########################DIY Pytorch train_ch3 function & Fashion-MNIST data load function############################################
import torch
from IPython import display
from d2l import torch as d2l



def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]



class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]




class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)




def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

from torchvision import transforms
import torchvision


def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像或numpy.ndarray转换为tensor，并归一化到[0,1]
    transforms.Normalize((0.5,), (0.5,))  # Fashion-MNIST的标准化参数
    #这是对图像数据进行标准化（归一化）处理的操作
    # 标准化处理：将图像数据转换为均值为0.5，标准差为0.5的标准正态分布
    ])
    # 加载Fashion-MNIST训练数据集
    mnist_train = torchvision.datasets.FashionMNIST(
        root='./data',      # 数据存储路径
        train=True,         # 指定为训练集
        transform=transform,  # 应用上面定义的预处理
        download=True       # 如果数据不存在则下载
        )

    # 加载Fashion-MNIST测试数据集
    mnist_test = torchvision.datasets.FashionMNIST(
        root='./data',      # 数据存储路径
        train=False,        # 指定为测试集
        transform=transform  # 应用相同的预处理
        )
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=4))

#########################################################################################################################


##########################################MLP function###################################################################

def mlp_function():
    pass




#########################################################################################################################