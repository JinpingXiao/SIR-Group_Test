import torch
import numpy as np
import matplotlib.pyplot as plt

class SGD_Momentum:
    def __init__(self, parameters, lr = 0.01, beta = 0.9, weight_decay = 0.01, nesterov = True):
       self.parameters = list(parameters) 
       self.lr = lr
       self.beta = beta
       self.weight_decay = weight_decay
       self.velocities = {id(p): torch.zeros_like(p.data) for p in self.parameters}
       self.nesterov = nesterov
    def step(self):
        with torch.no_grad():
            for p in self.parameters:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if self.weight_decay != 0:
                    d_p = d_p + self.weight_decay * p.data
                v_old = self.velocities[id(p)]
                v_new = self.beta * v_old + d_p
                if self.nesterov:
                    actural_update = d_p + self.beta * v_new
                else:
                    actural_update = v_new
                p.data -= self.lr * actural_update
                self.velocities[id(p)] = v_new 
    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

# 线性回归模型类
class LinearRegressionModel: # class LinearRegressionModel(nn.Module)
    def __init__(self, input_dim):
        self.weights = torch.randn(input_dim, 1, requires_grad=True)
        self.bias = torch.randn(1, requires_grad=True)
        '''
        self.weights = nn.Parameter(torch.randn(input_dim, 1))
        self.bias = nn.Parameter(torch.randn(1))
        '''
    
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
    num_samples = X.shape[0]
    for epoch in range(num_epochs):
        indices = torch.randperm(num_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        if epoch > 0 and epoch % 20 == 0:
            optimizer.lr *= 0.5
        y_pred = model.forward(X_shuffled)
        #y_pred = model(X)
        loss = loss_fn(y_pred, y_shuffled)
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
optimizer = SGD_Momentum(parameters, lr = 0.01, beta = 0.9)
#optimizer = SGD_Momentum(model.parameters(), lr = 0.01, beta = 0.9)