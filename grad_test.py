from torch import nn
import torch
from torch import optim

torch.manual_seed(10)
# 定义一个简单的网络
class net(nn.Module):
    def __init__(self, num_class=10):
        super(net, self).__init__()
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, num_class)

    def forward(self, x):
        return self.fc2(self.fc1(x))


model = net()


# 训练后的模型参数
print("model.fc1.weight", model.fc1.weight)
print("model.fc2.weight", model.fc2.weight)

for name, param in model.named_parameters():
    if "fc2" in name:
        param.requires_grad = False


# 训练后的模型参数
print("model.fc1.weight", model.fc1.weight)
print("model.fc2.weight", model.fc2.weight)

# 情况一：不冻结参数时
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)  # 传入的是所有的参数


x = torch.randn((3, 8))
label = torch.randint(0, 10, [3]).long()
output = model(x)

loss = loss_fn(output, label)

optimizer.zero_grad()
loss.backward()
optimizer.step()
print(model.fc1.weight.requires_grad)
print(model.fc1.weight.grad)
# print(model.fc2.weight.requires_grad)
# print(model.fc2.weight.grad)
# # print("model.fc1.weight", model.fc1.weight)
# # print("model.fc2.weight", model.fc2.weight)

