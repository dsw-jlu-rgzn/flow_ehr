"""
Flow Matching + Embedding 微调 框架
====================================
核心思路：
  - 小模型（Flow Matching Transformer）接收结构化 EHR 时序数据
  - 输出一个"病程压缩向量"（Patient Course Embedding）
  - 通过 Projector 映射到 LLM 的 embedding 空间，作为 soft prompt
  - LLM 在该压缩向量的引导下生成临床文书

三个训练阶段：
  Stage 1: 预训练 Flow Matching（无监督，学习 EHR 数据分布）
  Stage 2: 训练 Projector（连接 Flow Embedding → LLM 空间）
  Stage 3: 端到端微调（可选，联合微调 Projector + LLM LoRA）
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from tqdm import tqdm


# ============================================================
#  第一步：数据预处理 — 把 EHR 数据转为数值张量
# ============================================================

class EHRDataset(Dataset):
    """
    读取 data/AP/input/*.csv 和 data/DS/input/*.csv
    将每天的数值型数据（生命体征、化验）编码为向量序列
    """
    
    # 常用临床指标对应的关键词
    VITAL_SIGNS = ['heart rate', 'hr', 'blood pressure', 'bp', 'respiratory rate', 
                   'rr', 'temperature', 'temp', 'spo2', 'o2 sat']
    LABS = ['wbc', 'hemoglobin', 'hgb', 'hematocrit', 'platelet', 'sodium', 
            'potassium', 'creatinine', 'bun', 'glucose', 'lactate']
    
    def __init__(self, data_dir='data/AP/input', max_days=30, max_events_per_day=50):
        self.data_dir = data_dir
        self.max_days = max_days
        self.max_events_per_day = max_events_per_day
        self.samples = []
        self._load_data()
    
    def _parse_numeric_value(self, text):
        """从文本中提取数值，如 'HR: 88 bpm' → 88.0"""
        import re
        # 查找数字（包括小数）
        numbers = re.findall(r'(\d+\.?\d*)', text)
        return float(numbers[0]) if numbers else 0.0
    
    def _text_to_feature(self, text):
        """将一行EHR文本转为(是否数值型, 数值, 类别one-hot)"""
        text_lower = text.lower()
        is_vital = any(v in text_lower for v in self.VITAL_SIGNS)
        is_lab = any(l in text_lower for l in self.LABS)
        value = self._parse_numeric_value(text)
        # 类别编码: 0=用药/操作, 1=生命体征, 2=化验, 3=笔记
        if 'administer' in text_lower or 'infusion' in text_lower:
            category = 0
        elif is_vital:
            category = 1
        elif is_lab:
            category = 2
        else:
            category = 3
        return torch.tensor([value, category], dtype=torch.float32)
    
    def _load_data(self):
        import glob
        files = glob.glob(f'{self.data_dir}/*.csv')
        for fpath in files:
            df = pd.read_csv(fpath)
            if 'DAY' not in df.columns:
                # DS 数据没有 DAY 列，整段处理
                features = []
                for _, row in df.iterrows():
                    feat = self._text_to_feature(str(row.get('TEXT', '')))
                    features.append(feat)
                tensor = torch.stack(features) if features else torch.zeros((1, 2))
                self.samples.append(tensor)
            else:
                # AP 数据按天分组
                for day, day_df in df.groupby('DAY'):
                    day_features = []
                    for _, row in day_df.iterrows():
                        feat = self._text_to_feature(str(row.get('TEXT', '')))
                        day_features.append(feat)
                    if day_features:
                        self.samples.append({
                            'day': int(day),
                            'features': torch.stack(day_features),
                            'is_note': 1 if (day_df['IS_NOTE'] == 1).any() else 0
                        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if isinstance(sample, dict):
            return sample['features'], sample['is_note']
        return sample, 0  # DS 数据没有笔记标签


# ============================================================
#  第二步：Flow Matching 模型 — 病程压缩编码器
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Flow matching 的时间步 t 编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        # t: (batch,) 或 (batch, 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class CrossAttentionBlock(nn.Module):
    """用于时序数据融合的交叉注意力"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
    
    def forward(self, x, context=None):
        if context is not None:
            x = self.norm(x)
            x, _ = self.attn(x, context, context)
        return x


class FlowMatchingEncoder(nn.Module):
    """
    Flow Matching 病程编码器
    
    结构：
      EHR时序 → Transformer Encoder → 时间注意力池化 → 病程向量
    
    训练方式：conditional flow matching
      - 接收含噪数据 x_t = (1-t)*x_0 + t*ε
      - 预测速度场 v_θ(x_t, t, condition)
      - Loss = MSE(v_θ, 真实速度场 x_1 - x_0)
    """
    
    def __init__(self, input_dim=2, d_model=256, n_layers=6, n_heads=8, 
                 embedding_dim=128, max_days=30, max_events=50):
        super().__init__()
        
        self.d_model = d_model
        self.embedding_dim = embedding_dim
        self.max_days = max_days
        self.max_events = max_events
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_events, d_model) * 0.02)
        
        # 时间步编码
        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        # Transformer Encoder（Flow Matching 的 backbone）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 跨天融合（可处理多天的数据）
        self.day_attention = CrossAttentionBlock(d_model, n_heads)
        
        # 池化 → 病程向量
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, embedding_dim),
        )
        
        # 速度场预测头（Flow Matching 核心）
        self.velocity_head = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),  # 拼接 x_t 和 t_emb
            nn.GELU(),
            nn.Linear(d_model, input_dim),
        )
    
    def forward(self, x, t=None, return_embedding=True):
        """
        Args:
            x: (batch, seq_len, input_dim) EHR 数据
            t: (batch,) 时间步（flow matching 训练时使用）
            return_embedding: True返回病程向量，False返回预测速度场
        """
        batch_size, seq_len, _ = x.shape
        
        # 裁剪到 max_events
        x = x[:, :self.max_events, :]
        seq_len = x.size(1)
        
        # 输入投影 + 位置编码
        h = self.input_proj(x)  # (batch, seq_len, d_model)
        h = h + self.pos_encoding[:, :seq_len, :]
        
        if t is not None:
            # Flow Matching：拼接时间步编码
            t_emb = self.time_embed(t)        # (batch, d_model)
            t_emb = self.time_proj(t_emb)      # (batch, d_model)
            t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)
            h = h + t_emb
        
        # Transformer 编码
        h = self.transformer(h)  # (batch, seq_len, d_model)
        
        if return_embedding:
            # 返回病程压缩向量
            h_pool = h.permute(0, 2, 1)  # (batch, d_model, seq_len)
            embedding = self.pool(h_pool)  # (batch, embedding_dim)
            return embedding
        else:
            # 返回速度场预测（用于 flow matching 训练）
            h_flat = h.reshape(batch_size, -1)  # flatten
            t_emb = self.time_embed(t)
            velocity_input = torch.cat([h_flat, t_emb], dim=-1)
            velocity = self.velocity_head(velocity_input)
            # reshape back to (batch, seq_len, input_dim)
            velocity = velocity.view(batch_size, seq_len, -1)
            return velocity


# ============================================================
#  第三步：Flow Matching 训练（无监督预训练）
# ============================================================

def train_flow_matching(model, dataloader, epochs=100, lr=1e-4):
    """
    训练 Flow Matching 模型
    
    Flow Matching Loss:
      L = E_{t ~ U[0,1], x_0 ~ data, ε ~ N(0,1)} [ || v_θ(x_t, t) - (x_1 - x_0) ||^2 ]
    
    其中:
      x_t = (1-t) * x_0 + t * ε       # 线性插值噪声
      x_1 = ε                          # 纯噪声
      真实速度场 = x_1 - x_0 = ε - x_0  # 从数据流向噪声的方向
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Flow Matching Epoch {epoch+1}"):
            if isinstance(batch, (list, tuple)):
                x, _ = batch
            else:
                x = batch
            
            batch_size = x.size(0)
            seq_len = x.size(1)
            
            # 1. 随机采样时间步 t ∈ [0, 1]
            t = torch.rand(batch_size, device=x.device)
            
            # 2. 采样噪声 ε ~ N(0, 1)
            noise = torch.randn_like(x)
            
            # 3. 构造含噪数据 x_t = (1-t) * x_0 + t * ε
            t_expanded = t.view(-1, 1, 1).expand(-1, seq_len, x.size(-1))
            x_t = (1 - t_expanded) * x + t_expanded * noise
            
            # 4. 计算真实速度场（从数据流向噪声的方向）
            target_velocity = noise - x  # = x_1 - x_0
            
            # 5. 模型预测速度场
            pred_velocity = model(x_t, t, return_embedding=False)
            
            # 6. Flow Matching Loss (MSE)
            loss = F.mse_loss(pred_velocity, target_velocity)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(dataloader):.6f}")
    
    return model


# ============================================================
#  第四步：Projector — 将病程向量映射到 LLM 的 Embedding 空间
# ============================================================

class Projector(nn.Module):
    """
    将 Flow Matching 输出的病程向量 → LLM 的 Token Embedding 空间
    
    相当于"软提示生成器"（Soft Prompt Generator）
    类似于 LLaVA 的 vision↔language projector
    """
    
    def __init__(self, flow_dim=128, llm_dim=4096, num_soft_tokens=16):
        """
        Args:
            flow_dim: Flow Matching 输出的病程向量维度
            llm_dim: LLM 的 hidden_size（Mistral-7B = 4096）
            num_soft_tokens: 生成的 soft prompt token 数量
                更多 token = 更多信息传递给 LLM，但占用更多上下文长度
        """
        super().__init__()
        self.num_soft_tokens = num_soft_tokens
        
        # 将病程向量映射到 LLM embedding 空间
        # 输出形状: (batch, num_soft_tokens, llm_dim)
        self.projector = nn.Sequential(
            nn.Linear(flow_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(llm_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(llm_dim, num_soft_tokens * llm_dim),
        )
        
        # LayerNorm 稳定训练
        self.norm = nn.LayerNorm(llm_dim)
    
    def forward(self, flow_embedding):
        """
        Args:
            flow_embedding: (batch, flow_dim) 来自 Flow Matching Encoder
        Returns:
            soft_prompt: (batch, num_soft_tokens, llm_dim) 软提示
        """
        batch_size = flow_embedding.size(0)
        h = self.projector(flow_embedding)  # (batch, num_tokens * llm_dim)
        h = h.view(batch_size, self.num_soft_tokens, -1)  # (batch, num_tokens, llm_dim)
        h = self.norm(h)
        return h


# ============================================================
#  第五步：完整推理管线 — Flow Embedding → LLM 生成
# ============================================================

class FlowClinicalPipeline(nn.Module):
    """
    完整管线：
      EHR → FlowMatchingEncoder → 病程向量 → Projector → Soft Prompts → LLM
    
    在 LLM 中，soft prompts 作为前缀 token 插入到输入序列最前面
    """
    
    def __init__(self, flow_encoder, projector, llm, tokenizer, 
                 llm_hidden_size=4096):
        super().__init__()
        self.flow_encoder = flow_encoder
        self.projector = projector
        self.llm = llm          # frozen or LoRA fine-tuned
        self.tokenizer = tokenizer
        self.llm_hidden_size = llm_hidden_size
        
        # 将 flow_encoder 和 projector 设为可训练
        self.flow_encoder.requires_grad_(True)
        self.projector.requires_grad_(True)
    
    def get_soft_prompts(self, ehr_data):
        """
        将 EHR 数据转为 LLM 的 soft prompt tokens
        
        Args:
            ehr_data: (batch, seq_len, input_dim) 结构化 EHR
        Returns:
            soft_prompts: (batch, num_tokens, hidden_size) LLM 可读的嵌入
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            # 1. Flow Matching 编码 → 病程向量
            flow_embed = self.flow_encoder(ehr_data, return_embedding=True)
            # 2. Projector → LLM 空间的 soft prompts
            soft_prompts = self.projector(flow_embed)
        return soft_prompts
    
    def generate(self, ehr_data, instruction="", max_new_tokens=1000):
        """
        端到端生成临床文书
        
        流程：
          1. EHR → Flow Encoder → 病程向量
          2. Projector → Soft Prompts
          3. Soft Prompts + Instruction Tokenize → LLM Generate
        
        实现方式：修改 LLM 的 inputs_embeds，将 soft prompts 插入
        """
        self.eval()
        
        # Step 1: 获取 soft prompts
        soft_prompts = self.get_soft_prompts(ehr_data)  # (1, num_tokens, hidden_size)
        
        # Step 2: Tokenize instruction
        instruction_text = f"[INST] {instruction} [/INST]"
        instruction_tokens = self.tokenizer(
            instruction_text, return_tensors="pt", add_special_tokens=False
        )
        
        # Step 3: 获取 instruction 的 embedding
        with torch.no_grad():
            instruction_embeds = self.llm.get_input_embeddings()(
                instruction_tokens['input_ids'].to(ehr_data.device)
            )  # (1, instr_len, hidden_size)
        
        # Step 4: 拼接：soft_prompts + instruction_embeds
        combined_embeds = torch.cat([soft_prompts, instruction_embeds], dim=1)
        
        # Step 5: 设置 attention mask（扩展后的长度）
        attention_mask = torch.ones(
            (1, combined_embeds.size(1)), 
            device=ehr_data.device
        )
        
        # Step 6: 用 inputs_embeds 生成（跳过 embedding lookup）
        outputs = self.llm.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.3,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


# ============================================================
#  第六步：端到端训练（Stage 2 + Stage 3）
# ============================================================

def train_projector(pipeline, train_loader, val_loader, epochs=50, lr=5e-5):
    """
    Stage 2: 训练 Projector（冻结 LLM，训练 flow encoder + projector）
    
    Loss: 标准 CE Loss（让 LLM 在 soft prompt 引导下生成的文本接近真值）
    """
    
    # 冻结 LLM（只训练 flow_encoder 和 projector）
    for param in pipeline.llm.parameters():
        param.requires_grad_(False)
    pipeline.flow_encoder.requires_grad_(True)
    pipeline.projector.requires_grad_(True)
    
    optimizer = torch.optim.AdamW([
        {'params': pipeline.flow_encoder.parameters()},
        {'params': pipeline.projector.parameters()},
    ], lr=lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        pipeline.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Projector Training Epoch {epoch+1}"):
            # batch: (ehr_data, target_text)
            ehr_data, target_text = batch
            
            # 1. 获取 soft prompts
            soft_prompts = pipeline.get_soft_prompts(ehr_data)
            
            # 2. 拼接 instruction + target
            instruction = "你是一名ICU医生，请根据病程数据生成Assessment & Plan"
            instruction_tokens = pipeline.tokenizer(
                instruction, return_tensors="pt", add_special_tokens=False
            ).to(ehr_data.device)
            
            target_tokens = pipeline.tokenizer(
                target_text, return_tensors="pt", 
                padding='max_length', truncation=True, max_length=512
            ).to(ehr_data.device)
            
            # 3. 构建完整输入（soft_prompts + instruction + target）
            with torch.no_grad():
                instr_embeds = pipeline.llm.get_input_embeddings()(
                    instruction_tokens['input_ids']
                )
                target_embeds = pipeline.llm.get_input_embeddings()(
                    target_tokens['input_ids']
                )
            
            full_embeds = torch.cat([
                soft_prompts, instr_embeds, target_embeds
            ], dim=1)
            
            # 4. LLM forward（计算 loss 时只关注 target 部分）
            soft_len = soft_prompts.size(1)
            instr_len = instr_embeds.size(1)
            target_len = target_embeds.size(1)
            
            full_labels = torch.full_like(
                torch.zeros((1, soft_len + instr_len + target_len)), -100
            )
            full_labels[:, soft_len + instr_len:] = target_tokens['input_ids']
            
            outputs = pipeline.llm(
                inputs_embeds=full_embeds,
                labels=full_labels,
            )
            
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")
        
        # Validation
        val_loss = evaluate(pipeline, val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(pipeline.state_dict(), "best_pipeline.pt")
            print(f"  → New best model saved (val_loss={val_loss:.4f})")
    
    return pipeline


def evaluate(pipeline, dataloader):
    pipeline.eval()
    total_loss = 0
    with torch.no_grad():
        for ehr_data, target_text in dataloader:
            soft_prompts = pipeline.get_soft_prompts(ehr_data)
            
            instruction = "你是一名ICU医生..."
            instr_tokens = pipeline.tokenizer(
                instruction, return_tensors="pt", add_special_tokens=False
            ).to(ehr_data.device)
            
            target_tokens = pipeline.tokenizer(
                target_text, return_tensors="pt", 
                padding='max_length', truncation=True, max_length=512
            ).to(ehr_data.device)
            
            instr_embeds = pipeline.llm.get_input_embeddings()(instr_tokens['input_ids'])
            target_embeds = pipeline.llm.get_input_embeddings()(target_tokens['input_ids'])
            
            full_embeds = torch.cat([soft_prompts, instr_embeds, target_embeds], dim=1)
            
            soft_len = soft_prompts.size(1)
            instr_len = instr_embeds.size(1)
            full_labels = torch.full_like(
                torch.zeros((1, soft_len + instr_len + target_embeds.size(1))), -100
            )
            full_labels[:, soft_len + instr_len:] = target_tokens['input_ids']
            
            outputs = pipeline.llm(inputs_embeds=full_embeds, labels=full_labels)
            total_loss += outputs.loss.item()
    
    return total_loss / len(dataloader)


# ============================================================
#  第七步：主运行逻辑
# ============================================================

def build_complete_pipeline(device='cuda'):
    """
    构建完整的 Flow Matching + LLM 管线
    """
    
    # 1. Flow Matching 编码器（小模型，~50M 参数）
    flow_encoder = FlowMatchingEncoder(
        input_dim=2,        # (数值, 类别)
        d_model=256,        # Transformer hidden size
        n_layers=6,         # Transformer 层数
        n_heads=8,
        embedding_dim=128,  # 病程向量维度
    ).to(device)
    
    total_flow_params = sum(p.numel() for p in flow_encoder.parameters())
    print(f"Flow Encoder 参数量: {total_flow_params/1e6:.2f}M")
    
    # 2. Projector（病程向量 → LLM soft prompt）
    projector = Projector(
        flow_dim=128,
        llm_dim=4096,          # Mistral-7B hidden_size
        num_soft_tokens=16,    # 16 个 soft prompt tokens
    ).to(device)
    
    total_proj_params = sum(p.numel() for p in projector.parameters())
    print(f"Projector 参数量: {total_proj_params/1e6:.2f}M")
    
    # 3. LLM（冻结，用 LoRA 可选微调）
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    llm = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # 可选：给 LLM 加 LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
    )
    # llm = get_peft_model(llm, lora_config)  # 需要时取消注释
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 4. 组装完整管线
    pipeline = FlowClinicalPipeline(
        flow_encoder=flow_encoder,
        projector=projector,
        llm=llm,
        tokenizer=tokenizer,
    ).to(device)
    
    total_params = sum(p.numel() for p in pipeline.flow_encoder.parameters()) + \
                   sum(p.numel() for p in pipeline.projector.parameters())
    print(f"可训练参数量（不含LLM）: {total_params/1e6:.2f}M")
    
    return pipeline


if __name__ == '__main__':
    # ===== Stage 1: 预训练 Flow Matching =====
    print("=" * 50)
    print("Stage 1: Flow Matching 无监督预训练")
    print("=" * 50)
    
    dataset = EHRDataset(data_dir='data/AP/input')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, 
                           collate_fn=lambda batch: (
                               torch.nn.utils.rnn.pad_sequence(
                                   [b[0] for b in batch], batch_first=True
                               ),
                               torch.tensor([b[1] for b in batch])
                           ))
    
    flow_encoder = FlowMatchingEncoder().cuda()
    flow_encoder = train_flow_matching(flow_encoder, dataloader, epochs=50)
    torch.save(flow_encoder.state_dict(), "flow_encoder_pretrained.pt")
    
    # ===== Stage 2: 训练 Projector =====
    print("\n" + "=" * 50)
    print("Stage 2: Projector 训练（连接 Flow → LLM）")
    print("=" * 50)
    
    pipeline = build_complete_pipeline()
    # 加载预训练的 flow encoder
    pipeline.flow_encoder.load_state_dict(torch.load("flow_encoder_pretrained.pt"))
    
    # 这里需要准备带真值的配对数据
    # train_loader, val_loader = prepare_supervised_data()
    # pipeline = train_projector(pipeline, train_loader, val_loader)
    
    # ===== 推理演示 =====
    print("\n" + "=" * 50)
    print("推理演示")
    print("=" * 50)
    
    sample_ehr = torch.randn(1, 10, 2).cuda()  # 模拟 EHR 数据
    instruction = "根据患者的病程数据，生成今日的 Assessment & Plan"
    
    output = pipeline.generate(
        ehr_data=sample_ehr,
        instruction=instruction,
        max_new_tokens=500,
    )
    print(f"生成的临床文书:\n{output}")
