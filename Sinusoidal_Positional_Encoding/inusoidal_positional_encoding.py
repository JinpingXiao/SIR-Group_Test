import numpy as np

def get_sinusoidal_positional_encoding(max_len, d_model):
    """
    手动实现正弦位置编码
    :param max_len: 最大序列长度 (行数)
    :param d_model: 编码维度 (列数)
    :return: 形状为 (max_len, d_model) 的位置编码矩阵 (numpy.ndarray)
    """
    # 1. 初始化一个全零矩阵，用于存放最终的位置编码
    pe = np.zeros((max_len, d_model))
    
    # 2. 生成 pos 向量：[[0], [1], [2], ..., [max_len-1]]，形状为 (max_len, 1)
    position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
    
    # 3. 计算公式中的分母部分：10000^(2i / d_model)
    # 我们只需要计算 2i 即可，2i 的取值是 0, 2, 4, ..., 直到 d_model-2 (一共 d_model // 2 个)
    # 在代码中，为了数学上的计算简便和稳定性，通常利用自然对数 exp 和 log 来转换：
    # 10000^(2i/d_model) = exp( ln(10000^(2i/d_model)) ) = exp( (2i/d_model) * ln(10000) )
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model))
    
    # 4. 根据公式进行矩阵填充
    # position 形状是 (max_len, 1)，div_term 形状是 (d_model // 2, )
    # 相乘后利用广播机制，得到形状为 (max_len, d_model // 2) 的角度矩阵
    angle_matrix = position * div_term
    
    # 偶数列 (0, 2, 4, ...) 填充 sin 值
    pe[:, 0::2] = np.sin(angle_matrix)
    
    # 奇数列 (1, 3, 5, ...) 填充 cos 值
    pe[:, 1::2] = np.cos(angle_matrix)
    
    return pe

# --- 验证测试 ---
if __name__ == "__main__":
    # 假设句子最长为 10 个词，每个词的向量维度是 4 维
    max_len = 10
    d_model = 4
    
    pe_matrix = get_sinusoidal_positional_encoding(max_len, d_model)
    
    print(f"生成的矩阵形状: {pe_matrix.shape}")
    print("生成的位置编码矩阵内容:")
    # 打印结果，保留4位小数方便观察
    print(np.round(pe_matrix, 4))