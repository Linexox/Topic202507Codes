# HollaRec Conversation 任务数据处理指南

## 📋 数据需求概览

HollaRec 的 conversation 生成任务需要以下核心数据：

### 1. **输入数据（来自 Dataset）**

每个对话样本包含：

```python
{
    "role": "Recommender",              # 角色标识
    "context_tokens": [[tok1, tok2], [tok3, tok4], ...],  # 多轮历史对话
    "context_movies": [movie_id1, movie_id2, ...],         # 上下文提及的电影
    "response": [resp_tok1, resp_tok2, ...],               # 目标回复
    "user_id": int,                     # 用户ID
    "conv_id": int,                     # 对话ID
}
```

### 2. **输出数据（DataLoader 提供给模型）**

经过 `conv_batchify` 处理后的批次数据：

```python
{
    "context_tokens": Tensor[batch_size, context_len],    # Padded 上下文
    "context_movies": List[List[int]],                    # 电影ID列表
    "response": Tensor[batch_size, response_len],         # Padded 回复
    "user_id": List[int],                                 # 用户ID列表
    "conv_id": List[int],                                 # 对话ID列表
}
```

---

## 🔧 数据处理流程

### **Step 1: 数据过滤 (`conv_process_fn`)**

```python
def conv_process_fn(self):
    """
    只保留 Recommender 角色的样本
    因为 CRS 主要关注推荐系统的生成能力
    """
    dataset = []
    for conv in self.dataset:
        if conv["role"] == "Recommender":
            dataset.append(conv)
    return dataset
```

**作用**：从所有对话样本中筛选出系统（Recommender）需要生成回复的样本。

---

### **Step 2: 批处理 (`conv_batchify`)**

#### 2.1 **处理上下文 (Context Processing)**

```python
# 合并多轮对话
merged_context = merge_utt(
    conv['context_tokens'],              # [[utt1], [utt2], [utt3]]
    start_token_idx=self.start_token_idx,  # <s>
    split_token_idx=self.split_token_idx,  # <split>
    final_token_idx=self.end_token_idx     # </s>
)
# 结果: [<s>, utt1_tok1, ..., <split>, utt2_tok1, ..., <split>, utt3_tok1, ..., </s>]

# 截断到最大长度（保留最近的对话）
context = truncate(merged_context, self.context_truncate, truncate_tail=False)
```

**关键参数**：
- `truncate_tail=False`：从**头部**开始截断，保留最近的对话
- 典型配置：`context_truncate=512` 或 `1024`

#### 2.2 **处理回复 (Response Processing)**

```python
# 先截断
truncated_response = truncate(conv['response'], self.response_truncate - 2)

# 添加 start/end token
response = add_start_end_token_idx(
    truncated_response,
    start_token_idx=self.start_token_idx,  # <s>
    end_token_idx=self.end_token_idx       # </s>
)
# 结果: [<s>, resp_tok1, resp_tok2, ..., </s>]
```

**关键参数**：
- `-2`：为 `<s>` 和 `</s>` 预留空间
- 典型配置：`response_truncate=256`

#### 2.3 **Padding**

```python
# 上下文：左padding（保持最新对话在右侧）
context_tensor = padded_tensor(batch_context_tokens, pad_token_idx, pad_tail=False)
# 示例:
# [<pad>, <pad>, <s>, tok1, tok2, </s>]

# 回复：右padding（生成从左到右）
response_tensor = padded_tensor(batch_response, pad_token_idx, pad_tail=True)
# 示例:
# [<s>, tok1, tok2, </s>, <pad>, <pad>]
```

---

## 🎯 模型使用这些数据的方式

### 在 HypergraphLlava 模型中：

```python
def converse(self, batch, mode='train'):
    """
    对话生成
    
    Args:
        batch: {
            "context_tokens": Tensor[B, L_ctx],
            "context_movies": List[List[int]],
            "response": Tensor[B, L_resp],
            ...
        }
    """
    context = batch['context_tokens']        # [B, L_ctx]
    context_movies = batch['context_movies']  # List of movie IDs
    target = batch['response']                # [B, L_resp]
    
    # 1. 构建超图（基于 context_movies）
    hypergraphs = self.build_hypergraphs(context_movies)
    
    # 2. Encode context + hypergraph
    context_embeds = self.llava_model.encode_context(
        context, 
        hypergraph_data=hypergraphs
    )
    
    # 3. 生成回复
    if mode == 'train':
        # Teacher forcing
        loss = self.llava_model.compute_loss(context_embeds, target)
        preds = target  # 或者用模型预测的 token
        return loss, preds
    else:
        # Autoregressive generation
        preds = self.llava_model.generate(context_embeds, max_length=256)
        return preds
```

---

## 📊 数据流图示

```
Dataset Sample (raw)
├── context_tokens: [[utt1], [utt2], [utt3]]
├── context_movies: [1, 5, 10]
├── response: [tok1, tok2, ..., tokN]
└── role: "Recommender"

        ↓ conv_process_fn (filter)
        
Filtered Sample
└── Only "Recommender" samples

        ↓ conv_batchify
        
Batch (ready for model)
├── context_tokens: [B, max_ctx_len]
│   └── [<pad>, <pad>, <s>, utt1, <split>, utt2, <split>, utt3, </s>]
├── context_movies: [[1,5,10], [2,7], ...]
├── response: [B, max_resp_len]
│   └── [<s>, tok1, tok2, ..., tokN, </s>, <pad>, <pad>]
└── user_id, conv_id: metadata

        ↓ Model forward
        
Output
├── loss (train/valid)
└── generated_tokens (test)
```

---

## ⚙️ 配置建议

### 在 `config.yaml` 中设置：

```yaml
# Conversation 任务配置
context_truncate: 512      # 上下文最大长度
response_truncate: 256     # 回复最大长度

conv:
  batch_size: 32           # 对话任务批大小
  lr: 1e-4                 # 学习率
  epoch: 10                # 训练轮数
  
  # 生成配置
  generation:
    max_length: 256
    num_beams: 5           # Beam search
    temperature: 0.7
    top_p: 0.9
```

---

## 🔍 关键特点

### 1. **多轮对话合并**
- 使用 `<split>` token 分隔不同轮次
- 保证模型理解对话历史的上下文

### 2. **电影信息保留**
- `context_movies` 保持为 list 而非 tensor
- 模型内部根据这些 ID 构建超图

### 3. **Padding 策略**
- **Context**：左 padding → 保持最新对话在固定位置
- **Response**：右 padding → 生成时从左到右自然展开

### 4. **截断策略**
- **Context**：保留最近的对话（`truncate_tail=False`）
- **Response**：保留前面的内容（`truncate_tail=True`）

---

## 💡 与 Recommendation 任务的区别

| 维度 | Recommendation | Conversation |
|------|---------------|--------------|
| **目标** | 预测电影 ID | 生成自然语言回复 |
| **输入** | context + candidate movies | context + history |
| **输出** | movie scores | token sequence |
| **数据增强** | 每个电影一个样本 | 每个回复一个样本 |
| **Label** | movie ID (int) | response tokens (seq) |
| **Padding** | N/A | 需要 |

---

## 🚀 使用示例

```python
from crslab.data.dataloader.hollarec import HollaRecDataLoader

# 初始化
dataloader = HollaRecDataLoader(opt, dataset, vocab)

# 获取 conversation 数据
conv_dataloader = dataloader.get_conv_data(batch_size=32, shuffle=True)

# 遍历批次
for batch in conv_dataloader:
    # batch 已经包含所有必需数据
    context = batch['context_tokens']      # [32, 512]
    response = batch['response']            # [32, 256]
    movies = batch['context_movies']        # List[List[int]]
    
    # 传入模型
    loss, preds = model.converse(batch, mode='train')
```

---

## ✅ 检查清单

在实现 conversation 任务时，确保：

- [x] `conv_process_fn` 正确过滤 Recommender 样本
- [x] 上下文使用 `merge_utt` 合并多轮对话
- [x] 回复添加 `<s>` 和 `</s>` token
- [x] 截断长度配置合理（context: 512, response: 256）
- [x] Padding 方向正确（context: 左, response: 右）
- [x] 电影 ID 保持为 list 传递给模型
- [x] 返回的 tensor 已移动到正确的 device

---

**最后更新**: 2025-10-30
**作者**: HollaRec Team
