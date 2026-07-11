import torch
import numpy as np
import matplotlib.pyplot as plt
class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = {id(p): torch.zeros_like(p.data) for p in self.parameters}
        self.v = {id(p): torch.zeros_like(p.data) for p in self.parameters}
        self.t = {id(p): 0 for p in self.parameters}
    
    def step(self):
        with torch.no_grad():
            for p in self.parameters:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if self.weight_decay != 0:
                    d_p = d_p + self.weight_decay * p.data
                self.t[id(p)] += 1
                t = self.t[id(p)]

                m_old = self.m[id(p)]
                v_old = self.v[id(p)]

                m_new = self.beta1 * m_old + (1 - self.beta1) * d_p
                v_new = self.beta2 * v_old + (1 - self.beta2) * (d_p ** 2)

                m_hat = m_new / (1 - self.beta1 ** t)
                v_hat = v_new / (1 - self.beta2 ** t)
                
                p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
                self.m[id(p)] = m_new
                self.v[id(p)] = v_new
                
    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad = None
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
