import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

num_inputs = 784
num_hiddens = 256
num_outputs = 10
batch_size = 64
epochs = 15
lr = 0.1
x = np.arange(epochs)  # 用于将交叉熵损失可视化的横坐标
# 数据处理
train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

def my_ReLU(x):  # ReLU函数
    temp = torch.zeros_like(x)
    temp = torch.max(temp, x)
    return temp

class my_Classifier(nn.Module):
    # 初始化函数，对网路的输入层、隐含层、输出层的大小和使用的函数进行了规定。
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super().__init__()
        self.W1 = torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01
        self.b1 = torch.zeros((1, num_hiddens), requires_grad=True)
        self.W2 = torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01
        self.b2 = torch.zeros((1, num_outputs), requires_grad=True)
        self.a = torch.zeros(num_hiddens, requires_grad=True)
        self.B = torch.zeros(num_hiddens, requires_grad=True)
        self.y = torch.zeros(num_outputs, requires_grad=True)
        self.outputs = torch.zeros(num_outputs)

    def forward(self, x):  # 前向算法
        self.a = x @ self.W1 + torch.ones((batch_size, 1)) @ self.b1
        self.B = my_ReLU(self.a)
        self.y = self.B @ self.W2 + torch.ones((batch_size, 1)) @ self.b2
        self.outputs = F.softmax(self.y, dim=1)

        return self.outputs

    def backward(self, x, label):  # BP算法
        g = self.outputs - F.one_hot(label, num_classes=10)  #  采用独热编码使label转化为1 * 10的矩阵从而方便分类交叉熵的求导
        e = g @ self.W2.T
        e[self.a <= 0] = 0  # 此处为乘以ReLU函数的导数，小于等于0导数为0，大于0为导数为1

        # 以下ΔW2，Δb2，ΔW1，Δb1需除以batch_size进行正则化，防止梯度爆炸
        W2_change = self.B.T @ g / batch_size
        b2_change = torch.sum(g, dim=0) / batch_size
        W1_change = x.T @ e / batch_size
        b1_change = torch.sum(e, dim=0) / batch_size

        self.W2 = self.W2 - lr * W2_change
        self.b2 = self.b2 - lr * b2_change
        self.W1 = self.W1 - lr * W1_change
        self.b1 = self.b1 - lr * b1_change
        return x

print("自己手动实现FCNN的训练过程如下")
model1 = my_Classifier(num_inputs, num_hiddens, num_outputs)
func1 = nn.CrossEntropyLoss()
loss = 0
Loss1 = []  # 用于记录训练过程中的分类交叉熵损失
for epoch in range(epochs):
    for i, (feature, label) in enumerate(train_loader):
        if label.size(0) != batch_size:  # 由于数据集分批之后会存在最后一个批次为32个样本，故直接选择将其舍弃
            continue
        feature = feature.reshape(-1, num_inputs)
        outputs = model1.forward(feature)
        loss = func1(outputs, label)
        model1.backward(feature, label)
    print("Epoch%s损失为" % epoch, loss.item())
    Loss1.append(loss.item())
print("自己手动实现的FCNN模型训练完成")


# pytorch库实现
class Classifier_pytorch(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, num_outputs)
        )
    def forward(self, x):
        x = self.network(x)
        return x
print("调用pytorch库实现FCNN的训练过程如下")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2 = Classifier_pytorch(num_inputs, num_hiddens, num_outputs).to(device)
func2 = nn.CrossEntropyLoss()
optimizer = optim.SGD(model2.parameters(), lr=lr)
loss = 0
Loss2 = []  # 用于记录训练过程中的分类交叉熵损失
for epoch in range(epochs):
    for batch, (feature, label) in enumerate(train_loader):
        feature = feature.reshape(-1, num_inputs).to(device)
        label = label.to(device)
        outputs = model2(feature)
        loss = func2(outputs, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    Loss2.append(loss.cpu().item())
    print("Epoch%s损失为" % epoch, loss.item())
print("调用pytorch库实现的FCNN模型训练完成")


# 由于自己搭建的FCNN在cpu上训练，而pytorch搭建的FCNN在gpu上实现，所以预测过程两个分开处理

# 以下使用手动实现的FCNN和pytorch库实现的FCNN进行测试集上的预测并计算正确率
print("自己手动实现的FCNN进行预测")
correct = 0
total = 0
for i, (feature, label) in enumerate(test_loader):
    if label.size(0) != batch_size:
        continue
    feature = feature.reshape(-1, num_inputs)
    preds = model1.forward(feature)
    _, preds = torch.max(preds.data, 1)
    total = total + label.size(0)
    correct = correct + (preds == label).sum().item()
print("自己手动实现下，测试集上的正确率为: ", correct / total)

print("调用pytorch库实现的FCNN进行预测")
correct = 0
total = 0
for feature, label in test_loader:
    feature = feature.reshape(-1, num_inputs).to(device)
    label = label.to(device)
    preds = model2(feature)
    _, preds = torch.max(preds.data, 1)
    total = total + label.size(0)
    correct = correct + (preds == label).sum().item()
acc = correct / total
print("pytorch方法实现下，测试集上的正确率为: ", acc)

# 由于自己的模型无法直接保存参数，故手动实现保存
torch.save({
    'epoch': epochs,
    'learning_rate': lr,
    'batch_size': batch_size,
    'model_W1': model1.W1,
    'model_b1': model1.b1,
    'model_W2': model1.W2,
    'model_b2': model1.b2
}, "my_FCNN_model.pth")
torch.save(model2.state_dict(), "pytorch_FCNN_model.pth")  # 保存pytorch库实现的模型的参数

# 可视化分类交叉熵损失
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(x, Loss1, label='my_FCNN_acc')  # 自己实现训练过程中的损失
plt.plot(x, Loss2, label='pytorch_FCNN_acc')  # pytorch实现训练过程中的损失
plt.legend()
plt.show()
