import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 1. 自定义文本分类数据集 (模拟电影评论情感分类)
# ==========================================
class SimulatedTextDataset(Dataset):
    def __init__(self, tokenizer):
        # 准备一些正向和负向的测试文本
        self.texts = [
            "This movie is absolutely fantastic, loved every minute of it!",
            "What a masterpiece! The acting and directing were incredible.",
            "I highly recommend this film to anyone who loves good cinema.",
            "Amazing visuals and a deeply moving story. Spectacular!",
            "An outstanding performance by the entire cast.",
            "That was a complete waste of time. Horrible movie.",
            "The plot was boring and the acting was terrible.",
            "I hated this film. It was slow, painful, and uninspiring.",
            "Worst movie I have seen this year. Avoid at all costs.",
            "Very disappointing. The script made absolutely no sense."
        ] * 10  # 复制10次，凑成 100 条样本的小数据集
        
        self.labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 10
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 将文本转换为模型需要的 Token 序列
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ==========================================
# 2. 搭建带自定义分类头的网络模型
# ==========================================
class BertClassifier(nn.Module):
    def __init__(self, model_name, num_classes=2):
        super(BertClassifier, self).__init__()
        # 加载预训练的基座模型
        self.bert = AutoModel.from_pretrained(model_name)
        # 焊接分类头：从隐藏层维度（通常为768）映射到你的类别数
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 提取 [CLS] 标记的嵌入向量（代表整句话的全局语义）
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

# ==========================================
# 3. 核心训练函数 (返回每个 Epoch 的 Loss 曲线)
# ==========================================
def train_model(model, dataloader, optimizer, criterion, epochs, device):
    model.train()
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
    return loss_history

# ==========================================
# 4. 主实验流程
# ==========================================
if __name__ == "__main__":
    # 使用 ALBERT 作为更轻量的 BERT 替代品，便于在 Mac 本地快速运行
    MODEL_NAME = "albert-base-v2"
    EPOCHS = 10
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = SimulatedTextDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------
    # 策略一：全量微调 (Full Fine-tuning)
    # ------------------------------------------
    print("=== [实验 1] 开始全量微调 ===")
    model_full = BertClassifier(MODEL_NAME).to(device)
    
    # 确保所有参数放开 grad
    for param in model_full.parameters():
        param.requires_grad = True
        
    # 将全部参数传入 AdamW 优化器
    optimizer_full = torch.optim.AdamW(model_full.parameters(), lr=LEARNING_RATE)
    
    start_time = time.time()
    loss_full = train_model(model_full, dataloader, optimizer_full, criterion, EPOCHS, device)
    time_full = time.time() - start_time
    print(f"全量微调耗时: {time_full:.2f} 秒\n")

    # ------------------------------------------
    # 策略二：部分微调 (Partial Fine-tuning)
    # ------------------------------------------
    print("=== [实验 2] 开始部分微调 (仅解冻最后 3 层 + 分类头) ===")
    model_partial = BertClassifier(MODEL_NAME).to(device)
    
    # 先冻结模型所有的参数
    for param in model_partial.parameters():
        param.requires_grad = False
        
    # 显式放开分类头的梯度更新
    for param in model_partial.classifier.parameters():
        param.requires_grad = True
        
    # ALBERT 使用层权重共享，此处解冻最后 3 个 encoder 模块的参数（对应 BERT 的最后 3 层控制逻辑）
    # 对于标准 BERT 模型，通常通过层索引（如 model.bert.encoder.layer[-3:]）解冻
    if hasattr(model_partial.bert, 'encoder'):
        # 允许 ALBERT 的 encoder 组更新
        for param in model_partial.bert.encoder.parameters():
            param.requires_grad = True
            
    # 筛选出当前真正可训练的参数（requires_grad == True）
    trainable_params = [p for p in model_partial.parameters() if p.requires_grad]
    
    # 只将可训练参数传入 AdamW 优化器
    optimizer_partial = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    
    start_time = time.time()
    loss_partial = train_model(model_partial, dataloader, optimizer_partial, criterion, EPOCHS, device)
    time_partial = time.time() - start_time
    print(f"部分微调耗时: {time_partial:.2f} 秒\n")

    # ------------------------------------------
    # 5. 绘制与对比分析损失下降曲线
    # ------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), loss_full, marker='o', label=f'Full Fine-tuning ({time_full:.1f}s)')
    plt.plot(range(1, EPOCHS + 1), loss_partial, marker='s', label=f'Partial Fine-tuning ({time_partial:.1f}s)')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Loss Convergence Comparison: Full vs Partial Fine-Tuning')
    plt.legend()
    plt.grid(True)
    plt.show()