import torch
import matplotlib.pyplot as plt
import math

def sigmoid_and_grad(x):
    x = x.clone().detach().requires_grad_(True)
    y = 1 / (1 + torch.exp(-x))
    y.sum().backward()
    return y.detach(), x.grad.detach()

def gelu_and_grad(x):
    x = x.clone().detach().requires_grad_(True)
    y = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
    y.sum().backward()
    return y.detach(), x.grad.detach()

x = torch.linspace(-6, 6, 1000)

sig_y, sig_grad = sigmoid_and_grad(x)
gelu_y, gelu_grad = gelu_and_grad(x)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(x, sig_y, label='Sigmoid')
plt.plot(x, gelu_y, label='GELU')
plt.title('Function')
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.plot(x, sig_grad, label='Sigmoid Grad')
plt.plot(x, gelu_grad, label='GELU Grad')
plt.title('Gradient')
plt.legend()
plt.grid()

plt.show()

# 分析：
# Sigmoid梯度最大仅0.25，远离0迅速趋近0，多层链式相乘易梯度消失。
# GELU在零点附近保持较大的非零梯度(~0.5)，且变化平滑，反向传播更稳定。
# Transformer中LayerNorm使激活长期集中在0附近，因此大量神经元工作在GELU最有利的梯度区域。
# 同时GELU负区不硬截断(不同于ReLU)，保留弱负特征与连续梯度，更适合高维连续语义表示学习。