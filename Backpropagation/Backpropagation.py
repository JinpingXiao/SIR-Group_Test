import torch
import torch.nn as nn

# ==========================================
# 1. 严格使用你提供的原始初始化代码（完全未动）
# ==========================================
torch.manual_seed(0)

input_dim = 10
hidden_dim = 20
output_dim = 1

x = torch.randn(1, input_dim)
y_true = torch.randn(1, output_dim)

W1 = torch.randn(input_dim, hidden_dim, requires_grad=True)
b1 = torch.randn(1, hidden_dim, requires_grad=True)
W2 = torch.randn(hidden_dim, output_dim, requires_grad=True)
b2 = torch.randn(1, output_dim, requires_grad=True)

model_autograd = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim)
)
with torch.no_grad():
    model_autograd[0].weight.copy_(W1.T)
    model_autograd[0].bias.copy_(b1)
    model_autograd[2].weight.copy_(W2.T)
    model_autograd[2].bias.copy_(b2)


# ==========================================
# 2. 手动前向传播（使用独立的 W1, b1, W2, b2 变量）
# ==========================================
Z1 = torch.matmul(x, W1) + b1
A1 = torch.clamp(Z1, min=0)  # ReLU
Z2 = torch.matmul(A1, W2) + b2
loss_manual = torch.sum((Z2 - y_true) ** 2)


# ==========================================
# 3. 手动反向传播（精确推导）
# ==========================================
dZ2 = 2 * (Z2 - y_true)
dW2 = torch.matmul(A1.T, dZ2)
db2 = torch.sum(dZ2, dim=0, keepdim=True)

dA1 = torch.matmul(dZ2, W2.T)
dZ1 = dA1.clone()
dZ1[Z1 <= 0] = 0

dW1 = torch.matmul(x.T, dZ1)
db1 = torch.sum(dZ1, dim=0, keepdim=True)


# ==========================================
# 4. 使用你定义的 model_autograd 跑全自动求导
# ==========================================
# 前向传播
out_autograd = model_autograd(x)
loss_autograd = torch.sum((out_autograd - y_true) ** 2)

# 反向传播，此时梯度会注入到 model_autograd 的各层中
loss_autograd.backward()


# ==========================================
# 5. 梯度误差校验
# ==========================================
# 注意：因为 nn.Linear 内部的 weight 形状是 (out, in)，即 W.T
# 所以对比时，自动求导的梯度 model_autograd[0].weight.grad 必须转置 (.T) 才能对齐 dW1
diff_W1 = torch.max(torch.abs(dW1 - model_autograd[0].weight.grad.T))
diff_b1 = torch.max(torch.abs(db1 - model_autograd[0].bias.grad))
diff_W2 = torch.max(torch.abs(dW2 - model_autograd[2].weight.grad.T))
diff_b2 = torch.max(torch.abs(db2 - model_autograd[2].bias.grad))

print("--- 严格对齐你提供代码的梯度校验 ---")
print(f"W1 梯度最大绝对误差: {diff_W1.item():.2e}")
print(f"b1 梯度最大绝对误差: {diff_b1.item():.2e}")
print(f"W2 梯度最大绝对误差: {diff_W2.item():.2e}")
print(f"b2 梯度最大绝对误差: {diff_b2.item():.2e}")

# 自动化断言
assert diff_W1 < 1e-6, "W1 校验失败"
assert diff_b1 < 1e-6, "b1 校验失败"
assert diff_W2 < 1e-6, "W2 校验失败"
assert diff_b2 < 1e-6, "b2 校验失败"

print("\n完美通过！手动推导的梯度与 nn.Sequential 自动计算的梯度完全对齐。")