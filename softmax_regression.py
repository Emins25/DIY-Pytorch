# 导入必要的库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器模块
import torchvision  # 计算机视觉工具包
import torchvision.transforms as transforms  # 数据转换和预处理工具
from torch.utils.data import DataLoader  # 数据加载器

# 设置随机种子为42，确保每次运行结果一致
torch.manual_seed(42)

# 定义Fashion-MNIST数据集的类别标签
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 定义模型的超参数
input_size = 784    # 输入维度：28x28像素的图片展平后的大小
hidden_size1 = 256  # 第一个隐藏层的神经元数量
hidden_size2 = 128  # 第二个隐藏层的神经元数量
num_classes = 10    # 输出类别数量（10种服装类别）
num_epochs = 5      # 训练轮数
batch_size = 256    # 每批处理的样本数
learning_rate = 0.1  # 增大学习率，因为SGD通常需要更大的学习率
momentum = 0.9      # 添加动量参数，帮助SGD更好地收敛

# 定义数据预处理流程
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像或numpy.ndarray转换为tensor，并归一化到[0,1]
    transforms.Normalize((0.5,), (0.5,))  # Fashion-MNIST的标准化参数
])

# 加载Fashion-MNIST训练数据集
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data',      # 数据存储路径
    train=True,         # 指定为训练集
    transform=transform,  # 应用上面定义的预处理
    download=True       # 如果数据不存在则下载
)

# 加载Fashion-MNIST测试数据集
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data',      # 数据存储路径
    train=False,        # 指定为测试集
    transform=transform  # 应用相同的预处理
)

# 创建训练数据加载器
train_loader = DataLoader(
    dataset=train_dataset,  # 指定数据集
    batch_size=batch_size,  # 指定批次大小
    shuffle=True            # 随机打乱数据
)

# 创建测试数据加载器
test_loader = DataLoader(
    dataset=test_dataset,   # 指定数据集
    batch_size=batch_size,  # 指定批次大小
    shuffle=False           # 测试集不需要打乱
)

# 定义神经网络模型类
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()  # 调用父类的初始化方法
        self.flatten = nn.Flatten()  # 将2D图像（28x28）展平为1D向量（784）
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

# 创建模型实例
model = NeuralNet(input_size, hidden_size1, hidden_size2, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数（内部包含softmax）
optimizer = optim.SGD(
    model.parameters(),    # 模型参数
    lr=learning_rate,      # 学习率
    momentum=momentum      # 动量参数
)

# 定义训练函数
def train_model():
    model.train()  # 设置模型为训练模式
    total_step = len(train_loader)  # 计算每个epoch的总批次数
    
    for epoch in range(num_epochs):  # 遍历每个训练轮次
        for i, (images, labels) in enumerate(train_loader):  # 遍历每个批次
            # 前向传播
            outputs = model(images)  # 通过模型获取预测结果
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

# 定义测试函数
def test_model():
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 关闭梯度计算
        correct = 0  # 正确预测的样本数
        total = 0    # 总样本数
        class_correct = [0] * num_classes  # 每个类别的正确预测数
        class_total = [0] * num_classes    # 每个类别的总样本数
        
        for images, labels in test_loader:  # 遍历测试集
            outputs = model(images)  # 获取模型预测结果
            _, predicted = torch.max(outputs.data, 1)  # 获取最大概率的类别
            #outputs.data是输出结果，每一行对应一个样本在每一个类别上的概率
            #dim=1表示在行维度上取最大的值 
            #torch.max函数返回两个值，第一个值是最大值，第二个值是最大值的索引
            #predicted是最大概率的类别索引，所以最终的结果是返回一个包括所有最大值索引的向量
            total += labels.size(0)  # 累加样本数
            correct += (predicted == labels).sum().item()  # 累加正确预测数
            
            # 计算每个类别的准确率
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
        
        # 打印总体准确率
        print(f'总体准确率: {100 * correct / total:.2f}%')
        
        # 打印每个类别的准确率
        print('\n各类别准确率:')
        for i in range(num_classes):
            print(f'{classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')

# 程序入口
if __name__ == '__main__':
    print("开始训练...")
    train_model()  # 训练模型
    print("\n开始测试...")
    test_model()   # 测试模型
