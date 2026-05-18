import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """
    自注意力机制的单个注意力头 (One head of self-attention)
    负责从输入序列中提取特定子空间的特征表示。
    """

    def __init__(self, head_size):
        super().__init__()
        # 定义 Key (键) 线性层：为每个 token 生成用于被匹配的特征
        self.key = nn.Linear(n_embd, head_size, bias=False)
        # 定义 Query (查询) 线性层：为每个 token 生成用于寻找其他相关 token 的特征
        self.query = nn.Linear(n_embd, head_size, bias=False)
        # 定义 Value (值) 线性层：为每个 token 生成实际需要被提取的特征内容
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # 注册下三角掩码矩阵 (Mask)：确保在预测时，当前 token 只能看到自己及之前的 token，
        # 不能看到未来的 token（保证自回归模型的因果性）。
        # register_buffer 可确保其作为模型状态被保存，但不会作为参数被优化器更新。
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # Dropout 层：随机丢弃一些注意力权重，防止过拟合
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x 的输入形状: (B: batch_size, T: time-step/序列长度, C: n_embd/通道数)
        B, T, C = x.shape
        
        # 计算每个 token 的 Query 和 Key 向量
        q = self.query(x) # (B, T, head_size)
        k = self.key(x)   # (B, T, head_size)
        
        # 1. 计算注意力分数 ("affinities")
        # 将 q 和 k 转置后相乘，得到 token 之间的相关性矩阵
        # 乘以 C**-0.5 (缩放点积) 是为了防止点积结果过大，导致 softmax 梯度消失
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        
        # 应用掩码矩阵：将未来时间步对应的注意力分数设为负无穷大 (-inf)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        
        # 应用 softmax 将注意力分数转换为概率分布（每行的和为 1）
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        
        # 随机 Dropout 一些注意力连接
        wei = self.dropout(wei)
        
        # 2. 执行值的加权聚合 (weighted aggregation of the values)
        # 计算每个 token 的 Value 向量
        v = self.value(x) # (B, T, head_size)
        
        # 根据注意力权重矩阵 wei，对 Value 向量进行加权求和，得到最终的输出特征
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制 (Multiple heads of self-attention in parallel)
    将多个单头注意力 (Head) 并行运行，允许模型同时关注不同表示子空间的信息。
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        # 创建多个 Head 实例，并将它们存储在一个 ModuleList 中
        # num_heads 是头的数量，head_size 是每个头的特征维度
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        
        # 投影层 (Projection layer)：将多个头拼接后的结果再次映射回原始特征维度 (n_embd)
        # 这也是残差连接路径上的线性变换
        self.proj = nn.Linear(n_embd, n_embd)
        
        # Dropout 层：用于特征输出阶段的正则化，防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 遍历所有的注意力头，分别对输入 x 进行处理，并将结果在最后一个维度 (dim=-1，即特征维度) 上拼接起来
        # 每个头输出 (B, T, head_size)，拼接后变为 (B, T, num_heads * head_size) = (B, T, n_embd)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        
        # 通过线性投影层进行特征融合
        out = self.proj(out)
        
        # 应用 Dropout
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """
    前馈神经网络层 (A simple linear layer followed by a non-linearity)
    在自注意力机制完成 token 间的信息交流后，该层负责对每个 token 的特征进行独立的非线性变换计算。
    """
    def __init__(self, n_embd):
        super().__init__()
        # 使用 nn.Sequential 将多个层按顺序串联
        self.net = nn.Sequential(
            # 第一层线性变换：将特征维度放大 4 倍 (n_embd -> 4 * n_embd)
            # 这赋予了模型更大的表达能力去学习复杂的特征组合
            nn.Linear(n_embd, 4 * n_embd),
            # 非线性激活函数 ReLU：引入非线性，使得模型能够拟合复杂的函数
            nn.ReLU(),
            # 第二层线性变换 (投影层)：将维度从放大后的 4倍 映射回原始维度 (4 * n_embd -> n_embd)
            # 以便能够顺利进行后续的残差连接
            nn.Linear(4 * n_embd, n_embd),
            # Dropout 层：用于特征输出阶段的正则化，防止过拟合
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # 将输入数据 x 按顺序通过定义的网络层
        return self.net(x)

class Block(nn.Module):
    """
    Transformer 基础块 (Transformer block: communication followed by computation)
    包含两部分：多头自注意力机制（负责信息交流）和前馈神经网络（负责特征计算）。
    """
    def __init__(self, n_embd, num_heads):
        # n_embd: 嵌入维度，num_heads: 注意力头的数量
        super().__init__()
        # 计算每个头的维度大小
        head_size = n_embd // num_heads
        
        # 1. 实例化多头自注意力层 (Communication - 交流信息)
        self.sa = MultiHeadAttention(num_heads, head_size)
        
        # 2. 实例化前馈神经网络层 (Computation - 独立计算)
        self.ffn = FeedForward(n_embd)
        
        # 实例化两个层归一化 (Layer Normalization)
        # 在特征输入到注意力层或前馈网络之前对其进行标准化，使得训练更稳定
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-Norm 架构：先进行层归一化，再送入自注意力层，最后加上残差连接 (x + ...)
        # 残差连接 (Residual connection) 有助于解决深层网络中的梯度消失问题
        x = x + self.sa(self.ln1(x)) # Residual connection after self-attention
        
        # 同样，先归一化，再送入前馈网络，最后加上残差连接
        x = x + self.ffn(self.ln2(x)) # Residual connection after FFN
        return x

class GPTLanguageModel(nn.Module):
    """
    完整的 GPT 语言模型架构
    """
    def __init__(self):
        super().__init__()
        # 1. 词嵌入表 (Token Embedding Table)：将每个词的整数索引映射为一个高维向量 (n_embd)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        # 2. 位置嵌入表 (Position Embedding Table)：为了让模型感知序列中词的先后顺序
        # (因为注意力机制本身是忽略位置的集合运算)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # 3. Transformer Blocks 的堆叠：按顺序包含 n_layer 个 Block
        # 使用 nn.Sequential(*[...]) 将它们串联起来
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        
        # 4. 最终的层归一化 (Final Layer Normalization)
        self.ln = nn.LayerNorm(n_embd)
        
        # 5. 语言模型头 (Language Model Head)：最后的线性层，将特征向量映射回词汇表大小 (vocab_size)
        # 用于输出每个词对应的对数概率 (logits)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # 应用自定义的权重初始化，提升训练初期的稳定性
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # 初始化线性层和嵌入层的权重为均值为 0，标准差为 0.02 的正态分布
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # 偏置初始化为 0
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx: 当前上下文的词索引张量，形状为 (B, T)
        # targets: 目标词索引张量，通常是 idx 整体向后平移一位
        
        # 提取词嵌入特征 (B, T, C)
        tok_emb = self.token_embedding_table(idx) 
        
        # 提取位置嵌入特征：为 0 到 T-1 的位置生成对应的向量 (T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        
        # 将词嵌入和位置嵌入相加，得到包含位置信息的最终特征表示 (B, T, C)
        x = tok_emb + pos_emb 
        
        # 将特征送入所有 Transformer Blocks 进行深度处理 (B, T, C)
        x = self.blocks(x) 
        
        # 应用最后的层归一化 (B, T, C)
        x = self.ln(x) 
        
        # 通过线性层输出 logits (预测分数)，形状变为 (B, T, vocab_size)
        logits = self.lm_head(x) 

        # 如果没有传入目标标签 (targets)，则表示仅用于推理/生成，不计算损失
        if targets is None:
            loss = None
        else:
            # 在计算交叉熵损失之前，为了适配 PyTorch 的接口，需要将维度展平
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # 将 B 和 T 维度合并
            targets = targets.view(B*T)  # 同样展平 targets
            # 计算交叉熵损失
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # 文本生成函数：在给定初始上下文 idx 的基础上，连续生成 max_new_tokens 个新词
        # idx 是形状为 (B, T) 的整数张量
        for _ in range(max_new_tokens):
            # 因为模型有最大上下文长度 (block_size) 限制，所以截取最后 block_size 个 token 作为输入
            idx_cond = idx[:, -block_size:]
            
            # 调用 forward 函数，获取预测结果 (我们不需要 loss)
            logits, loss = self(idx_cond)
            
            # 我们只关心当前序列的"最后一个"时间步产生的预测结果
            logits = logits[:, -1, :] # 形状变为 (B, C)，即 (B, vocab_size)
            
            # 应用 softmax，将预测分数转化为概率分布
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            # 根据概率分布进行多项式采样 (Multinomial sampling)，得到下一个词的索引
            idx_next = torch.multinomial(probs, num_samples=1) # 形状为 (B, 1)
            
            # 将新生成的词拼接到历史序列的末尾，并继续循环
            idx = torch.cat((idx, idx_next), dim=1) # 形状变为 (B, T+1)
            
        return idx

model = GPTLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
