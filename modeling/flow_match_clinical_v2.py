"""
Flow Matching + 文本 Embedding 病程压缩框架 (v2)
=================================================
相比 v1 的核心改进：
  1. EHR 数据不转为数值向量 → 保留原始自然语言文本
  2. 用小文本编码器 (BERT/Sentence-T5) 将文本转为语义 embedding
  3. Flow Matching 在语义 embedding 空间做无监督学习
  4. 压缩后的病程向量通过 Projector 注入 LLM

数据流：
  EHR文本 → Sentence-BERT → 语义向量序列 → Flow Matching 压缩 → LLM生成
                                  ↑
                         Flow Matching 在这里做无监督训练
                         学习"临床文本的语义演化规律"
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    AutoModel  # 用于 Sentence-BERT
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import re


# ============================================================
#  第一步：文本编码器 — 将 EHR 自然语言转为语义向量
# ============================================================

class TextEmbeddingEncoder(nn.Module):
    """
    将自然语言的 EHR 文本转为固定维度的语义向量
    
    用 Sentence-BERT 或类似的小模型（~100M 参数）
    例如：all-MiniLM-L6-v2 (80M) 或 BGE-small-zh (30M)
    
    输出：每个事件一个 384 维向量，保留语义信息
    例如：
      "HR: 88 bpm"  →  [0.12, -0.34, 0.56, ...]  # 与"心率88"语义相近
      "WBC: 12.5"   →  [0.45, 0.12, -0.78, ...]  # 与"白细胞升高"语义相近
      "IV 抗生素"    →  [-0.23, 0.67, 0.01, ...]  # 与"用药"语义相近
    """
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', 
                 embed_dim=384, device='cuda'):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(device)
        # 冻结文本编码器（作为特征提取器）
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.encoder.eval()
    
    def forward(self, texts):
        """
        Args:
            texts: list[str] 或 list[list[str]]
                   单个文本或一组文本（一天的事件）
        Returns:
            embeddings: (batch, seq_len, embed_dim) 语义向量序列
        """
        if isinstance(texts[0], str):
            texts = [texts]
        
        batch_embeddings = []
        for batch_texts in texts:
            # 对每个文本编码
            encoded = self.tokenizer(
                batch_texts, padding=True, truncation=True, 
                max_length=128, return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.encoder(**encoded)
                # 取 [CLS] token 或 mean pooling
                embedding = outputs.last_hidden_state[:, 0, :]  # (seq_len, 384)
                embedding = F.normalize(embedding, p=2, dim=-1)  # L2归一化
            
            batch_embeddings.append(embedding)
        
        # 转成 batch 形式（用 padding 处理不等长）
        max_len = max(e.size(0) for e in batch_embeddings)
        padded = torch.zeros(len(batch_embeddings), max_len, self.embed_dim, 
                            device=self.device)
        for i, e in enumerate(batch_embeddings):
            padded[i, :e.size(0)] = e
        
        return padded


# ============================================================
#  第二步：数据集 — 将 EHR 数据按天组织为文本序列
# ============================================================

class NaturalLanguageEHRDataset(Dataset):
    """
    将 CSV 中的 EHR 数据按天组织为自然语言文本序列
    
    输出每个样本：
      - day_texts: list[str] 某一天的所有 EHR 事件文本
      - note_text: str (可选) 该天的医生笔记（用于 Stage 2 监督训练）
    """
    
    def __init__(self, data_dir='data/AP/input', return_notes=True):
        self.data_dir = data_dir
        self.return_notes = return_notes
        self.samples = []  # list of dict {day: int, texts: [str], note: str or None}
        self._load_data()
        print(f"Loaded {len(self.samples)} day-samples from {data_dir}")
    
    def _enrich_text(self, row):
        """
        对 EHR 文本做轻量增强：补充单位信息、规范化格式
        
        原文本: "HR is 88" 
        增强后: "Heart Rate is 88 bpm (normal range 60-100)"
        
        原文本: "12.5 mg Half Tablet of Metoprolol Tartrate is administered."
        增强后: "Medication: Metoprolol Tartrate 12.5 mg Half Tablet administered."
        """
        text = str(row.get('TEXT', ''))
        
        # 尝试提取数值，补充临床上下文（可选）
        # 这里保持原始文本，不做过多修改
        # 因为 Sentence-BERT 已经能理解"HR is 88"和"heart rate 88"的语义相似性
        
        # 但可以加前缀来帮助区分类型
        text_lower = text.lower()
        if any(v in text_lower for v in ['heart rate', 'hr', 'blood pressure', 
                                           'bp', 'respiratory rate', 'rr',
                                           'temperature', 'spo2', 'o2 sat']):
            text = f"[Vital] {text}"
        elif any(l in text_lower for l in ['wbc', 'hemoglobin', 'hgb', 'platelet',
                                            'sodium', 'potassium', 'creatinine',
                                            'glucose', 'lactate', 'ph']):
            text = f"[Lab] {text}"
        elif 'administer' in text_lower or 'infusion' in text_lower:
            text = f"[Medication] {text}"
        elif 'diagnosis' in text_lower or 'assessment' in text_lower:
            text = f"[Note] {text}"
        
        return text
    
    def _load_data(self):
        import glob
        files = sorted(glob.glob(f'{self.data_dir}/*.csv'))
        
        for fpath in files:
            df = pd.read_csv(fpath)
            if 'DAY' not in df.columns:
                # DS 数据：不分天，整体作为一段
                texts = [self._enrich_text(row) for _, row in df.iterrows()]
                self.samples.append({'day': 1, 'texts': texts, 'note': None})
            else:
                # AP 数据：按天分组
                for day, day_df in df.groupby('DAY'):
                    note_rows = day_df[day_df['IS_NOTE'] == 1]
                    data_rows = day_df[day_df['IS_NOTE'] == 0]
                    
                    day_texts = [self._enrich_text(row) for _, row in data_rows.iterrows()]
                    note_text = note_rows.iloc[-1]['TEXT'] if len(note_rows) > 0 else None
                    
                    if day_texts:  # 只保留有数据的天
                        self.samples.append({
                            'day': int(day),
                            'texts': day_texts,
                            'note': note_text
                        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.return_notes:
            return sample['texts'], sample['note']
        return sample['texts']


# ============================================================
#  第三步：Flow Matching 在语义空间（改进版）
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000., device=t.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class TextFlowMatchingEncoder(nn.Module):
    """
    Flow Matching 在语义空间做无监督学习
    
    输入：EHR 事件的语义向量序列 (由 Sentence-BERT 编码)
    输出：一个压缩的病程向量
    
    训练方式：Conditional Flow Matching（无监督）
      - 对语义向量加噪：x_t = (1-t) * x_0 + t * ε
      - 预测速度场：v_θ(x_t, t) → 学会"EHR语义的演化规律"
      - 训练后：编码器能提取出有意义的病程表示
    """
    
    def __init__(self, input_dim=384, d_model=256, n_layers=4, n_heads=8,
                 embedding_dim=128, max_seq_len=50):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # 输入投影（384 → 256）
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        
        # 时间步编码
        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 时序注意力池化 → 病程向量
        self.temporal_attention = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.temporal_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.to_embedding = nn.Sequential(
            nn.Linear(d_model, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        
        # Flow Matching 速度场预测头（无监督训练用）
        self.velocity_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, input_dim),  # 预测原始语义空间的速度
        )
    
    def forward(self, x, t=None, return_embedding=True):
        """
        Args:
            x: (batch, seq_len, input_dim) 语义向量序列
            t: (batch,) 时间步（flow matching 训练时）
            return_embedding: True→病程向量, False→速度场
        """
        batch_size, seq_len, _ = x.shape
        seq_len = min(seq_len, self.max_seq_len)
        x = x[:, :seq_len, :]
        
        # 投影到模型维度
        h = self.input_proj(x)  # (batch, seq_len, d_model)
        h = h + self.pos_encoding[:, :seq_len, :]
        
        if t is not None:
            t_emb = self.time_embed(t)
            t_emb = self.time_mlp(t_emb).unsqueeze(1)
            h = h + t_emb
        
        # Transformer
        h = self.transformer(h)  # (batch, seq_len, d_model)
        
        if return_embedding:
            # 时序注意力池化：用可学习的 query 从序列中提取关键信息
            query = self.temporal_query.expand(batch_size, -1, -1)
            pooled, attn_weights = self.temporal_attention(query, h, h)
            # pooled: (batch, 1, d_model)
            embedding = self.to_embedding(pooled.squeeze(1))  # (batch, embedding_dim)
            return embedding, attn_weights
        else:
            # 预测速度场返回输入空间
            velocity = self.velocity_head(h)  # (batch, seq_len, input_dim)
            return velocity


# ============================================================
#  第四步：Flow Matching 训练（无监督，语义空间）
# ============================================================

def train_text_flow_matching(flow_encoder, text_encoder, dataset, 
                              epochs=100, lr=1e-4, batch_size=16):
    """
    在语义空间中做 Flow Matching 无监督训练
    
    核心思想：
      1. 用 Sentence-BERT 把 EHR 文本转为语义向量 x
      2. 对 x 加噪：x_t = (1-t)*x + t*ε
      3. Flow Encoder 预测速度场
      4. Loss = MSE(预测速度场, 真实速度场)
    
    这个训练不需要任何标签！只需要 EHR 文本数据
    
    Flow Matching Loss 推导:
      x_0 ~ p_data (真实EHR语义分布)
      ε ~ N(0, I)
      x_t = (1-t) * x_0 + t * ε
      速度场 v(x_t, t) = d(x_t)/dt = ε - x_0
      Loss = E[||v_θ(x_t, t) - (ε - x_0)||²]
    """
    from torch.utils.data import DataLoader
    
    # 只取文本，不需要 notes
    train_dataset = NaturalLanguageEHRDataset(
        data_dir=dataset.data_dir, return_notes=False
    )
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                           collate_fn=lambda batch: batch)  # 保持为 list
    
    optimizer = torch.optim.AdamW(flow_encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    flow_encoder.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_texts in tqdm(dataloader, desc=f"Flow Matching Epoch {epoch+1}"):
            # batch_texts: list of list[str] (每个样本是当天的事件文本列表)
            
            # 1. 用 Sentence-BERT 把文本转为语义向量
            # 注意：这里每个样本是 list[str]，需要分别编码
            all_embeddings = []
            max_len = 0
            for day_texts in batch_texts:
                emb = text_encoder([day_texts])  # (1, seq_len, 384)
                seq_len = emb.size(1)
                max_len = max(max_len, seq_len)
                all_embeddings.append(emb)
            
            # pad 到相同长度
            batch_size_actual = len(all_embeddings)
            x = torch.zeros(batch_size_actual, max_len, text_encoder.embed_dim,
                          device=text_encoder.device)
            mask = torch.zeros(batch_size_actual, max_len, device=text_encoder.device)
            for i, emb in enumerate(all_embeddings):
                seq_len = emb.size(1)
                x[i, :seq_len] = emb
                mask[i, :seq_len] = 1
            
            # 2. 随机采样时间步 t
            t = torch.rand(batch_size_actual, device=x.device)
            
            # 3. 采样噪声
            noise = torch.randn_like(x)
            
            # 4. 构造含噪数据
            t_expanded = t.view(-1, 1, 1)
            x_t = (1 - t_expanded) * x + t_expanded * noise
            
            # 5. 真实速度场
            target_velocity = noise - x
            
            # 6. 模型预测速度场
            pred_velocity = flow_encoder(x_t, t, return_embedding=False)
            
            # 7. Loss（只对有效位置计算）
            loss = F.mse_loss(
                pred_velocity * mask.unsqueeze(-1),
                target_velocity * mask.unsqueeze(-1),
                reduction='sum'
            ) / mask.sum()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_encoder.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        print(f"Epoch {epoch+1}: FM Loss = {epoch_loss/num_batches:.6f}")
        
        # 每 10 个 epoch 用 t=0（纯数据）生成一次病程向量，看是否稳定
        if (epoch + 1) % 10 == 0:
            flow_encoder.eval()
            with torch.no_grad():
                sample_x = x[:1]  # 拿第一个样本
                embedding, attn = flow_encoder(sample_x, t=None, return_embedding=True)
                print(f"  Sample embedding norm: {embedding.norm().item():.4f}, "
                      f"Attention entropy: {(-(attn * torch.log(attn + 1e-8)).sum()).item():.4f}")
            flow_encoder.train()
    
    return flow_encoder


# ============================================================
#  第五步：Projector — 病程向量 → LLM Soft Prompt
# ============================================================

class Projector(nn.Module):
    """
    将病程向量（128维）映射到 LLM 的 embedding 空间（4096维）
    
    输出为多个 soft prompt token，包含更丰富的信息
    类似于 LLaVA 的 projector，但维度可配置
    """
    
    def __init__(self, flow_dim=128, llm_dim=4096, num_soft_tokens=16):
        super().__init__()
        self.num_soft_tokens = num_soft_tokens
        
        self.projector = nn.Sequential(
            nn.Linear(flow_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(llm_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(llm_dim, num_soft_tokens * llm_dim),
        )
        
        # 给每个 soft token 加独立的位置编码
        self.token_pos_encoding = nn.Parameter(
            torch.randn(1, num_soft_tokens, llm_dim) * 0.02
        )
        
        self.norm = nn.LayerNorm(llm_dim)
    
    def forward(self, flow_embedding):
        """
        Args:
            flow_embedding: (batch, flow_dim) 病程向量
        Returns:
            soft_prompts: (batch, num_soft_tokens, llm_dim)
        """
        batch_size = flow_embedding.size(0)
        h = self.projector(flow_embedding)
        h = h.view(batch_size, self.num_soft_tokens, -1)
        h = h + self.token_pos_encoding
        h = self.norm(h)
        return h


# ============================================================
#  第六步：完整管线 — 文本EHR → 软提示 → LLM生成
# ============================================================

class FlowClinicalPipelineV2(nn.Module):
    """
    完整的推理管线：
    
    EHR自然语言文本 → Sentence-BERT语义向量 
        → FlowMatchingEncoder病程向量 
        → Projector软提示 
        → LLM生成临床文书
    """
    
    def __init__(self, text_encoder, flow_encoder, projector, llm, tokenizer):
        super().__init__()
        self.text_encoder = text_encoder
        self.flow_encoder = flow_encoder
        self.projector = projector
        self.llm = llm
        self.tokenizer = tokenizer
    
    def get_soft_prompts(self, day_texts):
        """
        将一天的自然语言 EHR 文本转为 soft prompts
        
        Args:
            day_texts: list[str] 当天的事件文本列表
        Returns:
            soft_prompts: (1, num_soft_tokens, llm_dim)
            attn_weights: 时间注意力权重（用于可视化）
        """
        # 1. 文本 → 语义向量
        with torch.no_grad():
            embeddings = self.text_encoder([day_texts])  # (1, seq_len, 384)
        
        # 2. 语义向量 → 病程向量
        flow_embedding, attn_weights = self.flow_encoder(
            embeddings, t=None, return_embedding=True
        )  # (1, 128)
        
        # 3. 病程向量 → soft prompts
        soft_prompts = self.projector(flow_embedding)  # (1, num_tokens, 4096)
        
        return soft_prompts, attn_weights
    
    def generate(self, day_texts, instruction="", max_new_tokens=1000):
        """
        端到端生成
        
        Args:
            day_texts: list[str] 当天的 EHR 事件文本
            instruction: str 指令
            max_new_tokens: int 最大生成长度
        Returns:
            generated_text: str 生成的临床文书
            attn_weights: 时序注意力权重
        """
        self.eval()
        
        # 1. 获取 soft prompts
        soft_prompts, attn_weights = self.get_soft_prompts(day_texts)
        
        # 2. Tokenize instruction
        instruction_text = f"[INST] {instruction} [/INST]"
        instr_tokens = self.tokenizer(
            instruction_text, return_tensors="pt", add_special_tokens=False
        )
        
        # 3. 获取 instruction 的 embedding
        with torch.no_grad():
            instr_embeds = self.llm.get_input_embeddings()(
                instr_tokens['input_ids'].to(soft_prompts.device)
            )
        
        # 4. 拼接
        combined_embeds = torch.cat([soft_prompts, instr_embeds], dim=1)
        attention_mask = torch.ones(
            (1, combined_embeds.size(1)), device=soft_prompts.device
        )
        
        # 5. 生成
        outputs = self.llm.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.3,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text, attn_weights


# ============================================================
#  第七步：两阶段训练 + 推理演示
# ============================================================

def build_pipeline_v2(device='cuda'):
    """构建完整管线"""
    print("=" * 60)
    print("Building Flow Matching + LLM Pipeline (v2)")
    print("=" * 60)
    
    # 1. 文本编码器（冻结，~80M 参数）
    print("\n[1/4] Loading text encoder (Sentence-BERT)...")
    text_encoder = TextEmbeddingEncoder(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        embed_dim=384,
        device=device
    )
    print(f"  ✓ Text encoder loaded (embed_dim={text_encoder.embed_dim})")
    
    # 2. Flow Matching 编码器（可训练，~15M 参数）
    print("\n[2/4] Initializing Flow Matching encoder...")
    flow_encoder = TextFlowMatchingEncoder(
        input_dim=384,    # Sentence-BERT 输出维度
        d_model=256,      # Transformer hidden
        n_layers=4,       # 4 层 Transformer
        n_heads=8,
        embedding_dim=128,# 最终病程向量维度
        max_seq_len=50,   # 每天最多 50 个事件
    ).to(device)
    fm_params = sum(p.numel() for p in flow_encoder.parameters())
    print(f"  ✓ Flow Encoder: {fm_params/1e6:.2f}M parameters")
    
    # 3. Projector（可训练，~5M 参数）
    print("\n[3/4] Initializing Projector...")
    projector = Projector(
        flow_dim=128,
        llm_dim=4096,        # Mistral-7B hidden
        num_soft_tokens=16,
    ).to(device)
    proj_params = sum(p.numel() for p in projector.parameters())
    print(f"  ✓ Projector: {proj_params/1e6:.2f}M parameters")
    
    # 4. LLM（加载，4-bit 量化）
    print("\n[4/4] Loading LLM (Mistral-7B, 4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    llm = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  ✓ LLM loaded")
    
    trainable_params = fm_params + proj_params
    print(f"\n{'='*60}")
    print(f"Total trainable params: {trainable_params/1e6:.2f}M")
    print(f"LLM frozen: 7B parameters (4-bit quantized)")
    print(f"{'='*60}")
    
    pipeline = FlowClinicalPipelineV2(
        text_encoder=text_encoder,
        flow_encoder=flow_encoder,
        projector=projector,
        llm=llm,
        tokenizer=tokenizer,
    )
    
    return pipeline


def stage1_unsupervised_fm(pipeline, dataset, epochs=100):
    """Stage 1：无监督 Flow Matching 预训练"""
    print("\n" + "=" * 60)
    print("Stage 1: Unsupervised Flow Matching Pre-training")
    print("Training on EHR semantic space without any labels")
    print("=" * 60)
    
    pipeline.flow_encoder = train_text_flow_matching(
        pipeline.flow_encoder,
        pipeline.text_encoder,
        dataset,
        epochs=epochs,
        lr=1e-4,
        batch_size=8,
    )
    
    torch.save(pipeline.flow_encoder.state_dict(), "flow_encoder_semantic.pt")
    print(f"\n✓ Flow encoder saved to flow_encoder_semantic.pt")
    
    return pipeline


def stage2_supervised_projector(pipeline, train_loader, val_loader, epochs=50):
    """Stage 2：监督训练 Projector（冻结 Flow Encoder，微调 Projector）"""
    print("\n" + "=" * 60)
    print("Stage 2: Supervised Projector Training")
    print("Freeze Flow Encoder, train Projector with CE Loss")
    print("=" * 60)
    
    # 冻结 flow encoder
    for param in pipeline.flow_encoder.parameters():
        param.requires_grad_(False)
    pipeline.projector.requires_grad_(True)
    
    # 冻结 LLM
    for param in pipeline.llm.parameters():
        param.requires_grad_(False)
    
    optimizer = torch.optim.AdamW(
        pipeline.projector.parameters(), lr=5e-5
    )
    
    # 不加 LoRA 时，LLM 的 outputs.logits 不可用
    # 这里演示用 inputs_embeds 计算 loss 的方法
    # 实际使用时，建议加 LoRA 以得到 logits
    
    print("Stage 2 requires LLM with LoRA for gradient flow.")
    print("Run with: pipeline.llm = get_peft_model(pipeline.llm, lora_config)")
    
    return pipeline


def demo_inference(pipeline, sample_texts):
    """推理演示"""
    print("\n" + "=" * 60)
    print("Demo: End-to-End Inference")
    print("=" * 60)
    
    print(f"\nInput EHR (day with {len(sample_texts)} events):")
    for t in sample_texts[:3]:
        print(f"  • {t}")
    if len(sample_texts) > 3:
        print(f"  • ... and {len(sample_texts)-3} more events")
    
    instruction = """Generate the Assessment and Plan section of a clinical progress note based on the patient's EHR data.

Assessment: Summarize active problems and relevant comorbidities.
Plan: Organized by medical problem, detailing interventions and care strategies."""

    with torch.no_grad():
        generated, attn_weights = pipeline.generate(
            sample_texts, instruction=instruction, max_new_tokens=500
        )
    
    print(f"\nGenerated Assessment & Plan:\n{generated}")
    
    # 可视化注意力
    print(f"\nAttention weights shape: {attn_weights.shape}")
    print("  (shows which EHR events were most important for compression)")


# ============================================================
#  主入口
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=0, 
                       help='0=build+demo, 1=train FM, 2=train projector')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    # 构建管线
    pipeline = build_pipeline_v2()
    
    # 加载数据集
    dataset = NaturalLanguageEHRDataset(data_dir='data/AP/input', return_notes=True)
    
    if args.stage == 1:
        # 只做 Stage 1：无监督 Flow Matching
        pipeline = stage1_unsupervised_fm(pipeline, dataset, epochs=args.epochs)
    
    elif args.stage == 2:
        # 加载预训练 Flow Encoder，做 Projector 训练
        pipeline.flow_encoder.load_state_dict(
            torch.load("flow_encoder_semantic.pt")
        )
        pipeline = stage2_supervised_projector(pipeline, None, None)
    
    else:
        # Demo：用随机初始化的 Flow Encoder 直接推理
        print("\nUsing randomly initialized Flow Encoder (not trained yet)")
        print("Results will be poor - run Stage 1 first!\n")
        
        # 拿一个样本做演示
        sample = dataset[0]
        day_texts = sample[0]  # list[str]
        
        demo_inference(pipeline, day_texts)
