import torch
import numpy as np
import matplotlib.pyplot as plt
# 线性回归模型类
class LinearRegressionModel:
    def __init__(self, input_dim):
        self.weights = torch.randn(input_dim, 1, requires_grad=True)
        self.bias = torch.randn(1, requires_grad=True)
    def forward(self, x):
        return torch.matmul(x, self.weights) + self.bias
    
# 数据生成函数
def generate_data(num_samples=100, input_dim=2):
    X = torch.randn(num_samples, input_dim)
    true_w = torch.tensor([[2.0], [-3.0]])
    true_b = torch.tensor([1.0])
    y = torch.matmul(X, true_w) + true_b + torch.randn(num_samples, 1) * 0.5
    return X, y

# 训练循环框架
def train_loop(optimizer, model, X, y, loss_fn, num_epochs=100):
    losses = []
    for epoch in range(num_epochs):
        y_pred = model.forward(X)
        #y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return losses

# 数据准备
X, y = generate_data()
model = LinearRegressionModel(X.shape[1])
parameters = [model.weights, model.bias]
loss_fn = torch.nn.MSELoss()
