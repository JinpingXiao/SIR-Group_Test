import torch

def lstm_cell_forward(x_t, h_prev, c_prev, W_gates, b_gates, hidden_dim):
    """手动实现单步 LSTM 单元的前向传播

    参数:
    x_t        -- 当前时间步输入, 形状为 (batch_size, input_dim)
    h_prev     -- 前一时刻隐藏状态, 形状为 (batch_size, hidden_dim)
    c_prev     -- 前一时刻细胞状态, 形状为 (batch_size, hidden_dim)
    W_gates    -- 合并的权重矩阵, 形状为 (input_dim + hidden_dim, 4 * hidden_dim)
    b_gates    -- 合并的偏置向量, 形状为 (4 * hidden_dim)
    hidden_dim -- 隐藏层维度 (int)

    返回:
    h_t        -- 更新后的隐藏状态, 形状为 (batch_size, hidden_dim)
    c_t        -- 更新后的细胞状态, 形状为 (batch_size, hidden_dim)
    """
    # 1. 拼接当前输入 x_t 和上一时刻隐藏状态 h_prev
    # 拼接后的形状: (batch_size, input_dim + hidden_dim)
    concat_input = torch.cat([h_prev, x_t], dim=1)

    # 2. 执行线性变换 (大矩阵乘法)
    # gates_linear 形状: (batch_size, 4 * hidden_dim)
    gates_linear = torch.matmul(concat_input, W_gates) + b_gates

    # 3. 将计算结果在最后一个维度(dim=1)切分为 4 块，分别对应三个门和一个候选状态
    # 每块的形状都是: (batch_size, hidden_dim)
    # 注：切分顺序需要与后续激活函数的应用一一对应，这里假设顺序为: 遗忘门, 输入门, 候选状态, 输出门
    f_linear, i_linear, g_linear, o_linear = torch.chunk(
        gates_linear, chunks=4, dim=1
    )

    # 4. 应用对应的激活函数
    # 门控使用 Sigmoid 限制在 (0, 1)
    f_t = torch.sigmoid(f_linear)  # 遗忘门
    i_t = torch.sigmoid(i_linear)  # 输入门
    o_t = torch.sigmoid(o_linear)  # 输出门

    # 候选细胞状态（即公式中的 \tilde{C}_t）使用 Tanh 限制在 (-1, 1)
    c_tilde_t = torch.tanh(g_linear)

    # 5. 更新细胞状态 c_t
    # c_t = f_t * c_prev + i_t * c_tilde_t (这里的 * 即为哈达马乘积 \odot)
    c_t = f_t * c_prev + i_t * c_tilde_t

    # 6. 更新隐藏状态 h_t
    # h_t = o_t * tanh(c_t)
    h_t = o_t * torch.tanh(c_t)

    return h_t, c_t

# 假设输入维度为10，隐藏维度为20，批处理大小为4
input_dim = 10
hidden_dim = 20
batch_size = 4

# 随机生成一个批次的输入张量
x_t = torch.randn(batch_size, input_dim)

# 假设初始隐藏状态和细胞状态都为零
h_prev = torch.zeros(batch_size, hidden_dim)
c_prev = torch.zeros(batch_size, hidden_dim)

# LSTM单元的权重和偏置
# 这里的权重和偏置是简化的，将四个门的权重和偏置合并为一个大张量，以减少手动拼接
W_gates = torch.randn(input_dim + hidden_dim, 4 * hidden_dim)
b_gates = torch.randn(4 * hidden_dim)

# 2. 调用手动实现的前向传播函数
h_t, c_t = lstm_cell_forward(
    x_t, h_prev, c_prev, W_gates, b_gates, hidden_dim
)

# 3. 使用 assert 语句验证输出张量的形状是否正确
expected_shape = (batch_size, hidden_dim)

assert (
    h_t.shape == expected_shape
), f"h_t 形状错误！期望 {expected_shape}，实际得到 {h_t.shape}"
assert (
    c_t.shape == expected_shape
), f"c_t 形状错误！期望 {expected_shape}，实际得到 {c_t.shape}"

print("[验证成功] 所有的张量形状（Shape）均完全符合预期！")
print(f"h_t 的形状: {h_t.shape}")
print(f"c_t 的形状: {c_t.shape}")