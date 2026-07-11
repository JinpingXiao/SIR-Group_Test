import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, key_padding_mask=None):
    """
    带掩码的缩放点积注意力机制实现
    
    参数:
    Q: 查询张量, 形状为 (batch_size, seq_len, d_k)
    K: 键张量, 形状为 (batch_size, seq_len, d_k)
    V: 值张量, 形状为 (batch_size, seq_len, d_k)
    key_padding_mask: 掩码张量, 形状为 (batch_size, seq_len), 
                      其中 True (或 1) 表示该位置是 padding，需要被遮掩
    """
    # 获取特征维度 d_k 用于缩放
    d_k = Q.size(-1)
    
    # 1. 计算注意力分数 (Scores = Q * K^T)
    # K.transpose(-2, -1) 将形状从 (batch_size, seq_len, d_k) 变成 (batch_size, d_k, seq_len)
    # torch.bmm 是批矩阵乘法 (batch matrix multiplication)
    scores = torch.bmm(Q, K.transpose(-2, -1))
    
    # 2. 缩放分数
    scores = scores / (d_k ** 0.5)
    
    # 3. 集成掩码功能
    if key_padding_mask is not None:
        # key_padding_mask 的形状是 (batch_size, seq_len_k)
        # 为了应用到 scores (batch_size, seq_len_q, seq_len_k) 上，
        # 我们需要将其维度扩展为 (batch_size, 1, seq_len_k)，PyTorch 会自动通过广播机制机制匹配
        mask = key_padding_mask.unsqueeze(1)
        
        # 使用 masked_fill_ 方法，将 mask 为 True 的位置替换为 -1e9
        scores = scores.masked_fill(mask, -1e9)
        
    # 4. 经过 Softmax 得到注意力权重 (Weights)
    # 在最后一个维度 (seq_len_k 维度) 上做归一化
    attention_weights = F.softmax(scores, dim=-1)
    
    # 5. 加权求和得到最终输出 (Output = Weights * V)
    output = torch.bmm(attention_weights, V)
    
    return output, attention_weights

# ==================== 验证环节 ====================

# 设定随机种子以确保结果可复现
torch.manual_seed(42)

d_k = 64
seq_len = 8
batch_size = 2

# 随机生成 Q, K, V 张量
Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_k)

# 掩码定义：1 (True) 代表需要被 mask 的位置
key_padding_mask = torch.tensor([
    [0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1]
], dtype=torch.bool)

# 运行实现的注意力函数
output, weights = scaled_dot_product_attention(Q, K, V, key_padding_mask)

# 打印结果进行验证
print("=== 注意力权重矩阵 (Attention Weights) ===")
for i in range(batch_size):
    print(f"\n序列 {i+1} 的权重矩阵 (形状: {weights[i].shape}):")
    # 为了方便观察，我们打印前 2 个 Token 关注整句话的权重分布
    print(weights[i][:2])