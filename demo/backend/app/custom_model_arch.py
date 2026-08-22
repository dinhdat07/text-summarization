import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from torch.cuda.amp import autocast

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
  d_k = query.size(-1)
  scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
  if mask is not None:
    min_value = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(mask == 0, min_value)
  p_attn = F.softmax(scores, dim=-1)
  if dropout is not None:
    p_attn = dropout(p_attn)
  return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
  def __init__(self, h, d_model, dropout=0.1):
    super().__init__()
    assert d_model % h == 0
    self.d_k = d_model // h
    self.h = h
    self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
    self.dropout = nn.Dropout(dropout)

  def forward(self, query, key, value, mask=None, past_key_value=None, use_cache=False, is_cross_attention=False):
    batch_size = query.size(0)
    # batch_size, h, seq_len, d_k
    if past_key_value is not None:
        if is_cross_attention: 
            query = self.linears[0](query).view(batch_size, -1, self.h, self.d_k).transpose(1,2)
            key, value = past_key_value
        else: 
            query, key, value = [linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1,2) for linear, x in zip(self.linears[:3], (query, key, value))]
            K, V = past_key_value
            key = torch.cat([K, key], dim=-2)
            value = torch.cat([V, value], dim=-2)
    else: 
        query, key, value = [linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1,2) for linear, x in zip(self.linears[:3], (query, key, value))]
        
    present_key_value = (key, value) if use_cache else None

    if mask is not None:
        mask = mask.unsqueeze(1)
    x, attn = scaled_dot_product_attention(query, key, value, mask=mask, dropout=self.dropout)
    x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

    return self.linears[-1](x), present_key_value

class FeedForward(nn.Module):
  def __init__(self, d_model, d_ff, dropout=0.1):
    super().__init__()
    self.linear_1 = nn.Linear(d_model, d_ff)
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(d_ff, d_model)
    self.activation = nn.ReLU()

  def forward(self, x):
    return self.linear_2(self.dropout(self.activation(self.linear_1(x))))


class LayerNorm(nn.Module):
  def __init__(self, d_model, eps=1e-6):
    super().__init__()
    self.a_2 = nn.Parameter(torch.ones(d_model))
    self.b_2 = nn.Parameter(torch.zeros(d_model))
    self.eps = eps

  def forward(self, x):
    x_f32 = x.float()
    mean = torch.mean(x_f32, dim=-1, keepdim=True)
    var = torch.var(x_f32, dim=-1, keepdim=True, unbiased=False)
      
    out = (x_f32 - mean) / torch.sqrt(var + self.eps)
    return (out.to(x.dtype)) * self.a_2 + self.b_2


class ResidualConnection(nn.Module):
  def __init__(self, d_model, dropout=0.1):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.layer_norm = LayerNorm(d_model)

  def forward(self, x, sublayer):
    return x + self.dropout(sublayer(self.layer_norm(x)))

def clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class EncoderLayer(nn.Module):
  def __init__(self, d_model, d_ff, h, dropout=0.1):
    super().__init__()
    self.self_attn = MultiHeadAttention(h, d_model, dropout=dropout)
    self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)
    self.sublayer = clones(ResidualConnection(d_model, dropout=dropout), 2)
    self.d_model = d_model

  def forward(self, x, mask):
    x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask=mask)[0])
    x = self.sublayer[1](x, self.feed_forward)
    return x

class Encoder(nn.Module):
  def __init__(self, d_model, d_ff, h, N, dropout=0.1):
    super().__init__()
    self.layers = clones(EncoderLayer(d_model, d_ff, h, dropout=dropout), N)
    self.embedding = Embeddings(d_model, 36000)
    self.pe = PositionalEncoding(d_model, dropout=dropout)

  def forward(self, x, mask=None):
    x = self.embedding(x)
    x = self.pe(x)
    for layer in self.layers:
      x = layer(x, mask)
    return x

class DecoderLayer(nn.Module):
  def __init__(self, d_model, d_ff, h, dropout=0.1):
    super().__init__()
    self.masked_attn = MultiHeadAttention(h, d_model, dropout=dropout)
    self.attn = MultiHeadAttention(h, d_model, dropout=dropout)
    self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)
    self.sublayer = clones(ResidualConnection(d_model, dropout=dropout), 3)
    self.d_model = d_model

  def forward(self, x, memory, src_mask, tgt_mask, past_kv=None, use_cache=False):
    past_self_kv = past_kv[0] if past_kv is not None else None
    past_cross_kv = past_kv[1] if past_kv is not None else None        
    present_self_kv = None
    def self_attn_wrapper(q): 
        nonlocal present_self_kv 
        out, present_self_kv = self.masked_attn(q, q, q, mask=tgt_mask, past_key_value=past_self_kv, use_cache=use_cache, is_cross_attention=False)
        return out
    x = self.sublayer[0](x, self_attn_wrapper)

    present_cross_kv = None
    def cross_attn_wrapper(q): 
        nonlocal present_cross_kv
        out, present_cross_kv = self.attn(q, memory, memory, mask=src_mask, past_key_value=past_cross_kv, use_cache=use_cache, is_cross_attention=True)
        return out
    x = self.sublayer[1](x, cross_attn_wrapper)
    x = self.sublayer[2](x, self.feed_forward)
    return x, (present_self_kv, present_cross_kv)

class Decoder(nn.Module):
  def __init__(self, d_model, d_ff, h, N, dropout=0.1):
    super().__init__()
    self.layers = clones(DecoderLayer(d_model, d_ff, h, dropout=dropout), N)
    self.embedding = Embeddings(d_model, 36000)
    self.pe = PositionalEncoding(d_model, dropout=dropout)
    self.linear = nn.Linear(d_model, 36000)

  def forward(self, x, memory, src_mask, tgt_mask, past_kvs=None, use_cache=False, step=0):
    x = self.embedding(x)
    x = self.pe(x, step=step)
    present_kvs = [] 
    for i, layer in enumerate(self.layers):
      past_kv = past_kvs[i] if past_kvs is not None else None
      x, present_kv = layer(x, memory, src_mask, tgt_mask, past_kv=past_kv, use_cache=use_cache)
      present_kvs.append(present_kv)
    return self.linear(x), present_kvs

class BaselineTransformer(nn.Module):
  def __init__(self, d_model, d_ff, h, N, dropout=0.1):
    super().__init__()
    self.encoder = Encoder(d_model, d_ff, h, N, dropout=dropout)
    self.decoder = Decoder(d_model, d_ff, h, N, dropout=dropout)
    self.d_model = d_model

  def forward(self, src, tgt, src_mask=None, tgt_mask=None):
    memory = self.encoder(src, src_mask)
    output = self.decoder(tgt, memory, src_mask, tgt_mask)[0]
    return output

  @torch.inference_mode()
  def generate_summary(self, src_token, src_mask, bos_idx, eos_idx, pad_idx, max_len=360, strategy='greedy',penalty_tensor=None, base_penalty=0.0, **kwarg):
    device = next(self.parameters()).device
    device_type=device.type
    src_token = src_token.to(device)
    src_mask = src_mask.to(device)
    with torch.autocast(device_type=device_type, dtype=(torch.float16 if device_type == 'cuda' else torch.bfloat16), enabled=(device_type == 'cuda')):
      memory = self.encoder(src_token, src_mask)

    if strategy == 'greedy': 
        return self._greedy_search(src_token, src_mask, memory, bos_idx, eos_idx, pad_idx, max_len)
    elif strategy == 'greedy_with_penalty': 
        return self._greedy_search(src_token, src_mask, memory, bos_idx, eos_idx, pad_idx, max_len, penalty_tensor, base_penalty=base_penalty)

  def _greedy_search(self, src_token, src_mask, memory, bos_idx, eos_idx, pad_idx, max_len, penalty_tensor=None, base_penalty=0.0): 
      device = next(self.parameters()).device
      device_type=device.type
      batch_size = src_token.size(0)
      
      tgt_tokens = torch.full((batch_size,1), bos_idx, dtype=torch.long, device=device)
      unfinished = torch.ones((batch_size,1), dtype=torch.bool, device=device)

      past_kvs = None

      if penalty_tensor is not None:
          # penaldy_tensor shape: 1,vocab_size
          penalty_tensor = penalty_tensor.to(device)
          vocab_size = penalty_tensor.size(1)
          counts = torch.zeros((batch_size, vocab_size), dtype=torch.long, device=device)
      
      for step in range(max_len): 
          with torch.autocast(device_type=device_type, dtype=(torch.float16 if device_type == 'cuda' else torch.bfloat16), enabled=(device_type=='cuda')): 
              input_token = tgt_tokens if step == 0 else tgt_tokens[:, -1:]
              output, present_kvs = self.decoder(input_token, memory, src_mask, None, past_kvs=past_kvs, use_cache=True, step=step)
          next_token_logits = output[:, -1, :] # batch_size, vocab_size
          if penalty_tensor is not None: 
              mask = counts > 0
              next_token_logits = next_token_logits - (penalty_tensor * counts + mask * base_penalty)
          next_token = next_token_logits.argmax(dim= -1).unsqueeze(-1) # batch_size, 1
          next_token = next_token * unfinished + (~unfinished) * pad_idx
          tgt_tokens = torch.cat([tgt_tokens, next_token], dim=-1)
          unfinished = unfinished & (eos_idx != next_token)
          past_kvs = present_kvs
          if penalty_tensor is not None: 
              batch_indices = torch.arange(batch_size, device=device)
              active_mask = unfinished.squeeze(-1)
              active_indices = batch_indices[active_mask]
              active_tokens = next_token.squeeze(-1)[active_mask]
              counts[active_indices, active_tokens] += 1
          if unfinished.max() == 0: 
              break
      return tgt_tokens
              
              


# In[11]:


# Count for create histogram
from tqdm import tqdm
def generate_penalty(alpha=1.0):
    _dataloader = DataLoader(train_dataset, 
                              batch_size=BATCH_SIZE, 
                              shuffle=False, 
                              collate_fn=collate_fn,
                              pin_memory=False, 
                              num_workers=0)
    VOCAB_SIZE = 36000
    global_counts = torch.zeros(VOCAB_SIZE, dtype=torch.long, device='cpu')
    
    for article_ids, summary_ids, _, _ in tqdm(_dataloader): 
        text_ids = torch.cat([article_ids, summary_ids], dim=-1)
        tokens = text_ids.view(-1).cpu()
        batch_counts = torch.bincount(tokens, minlength=VOCAB_SIZE)
        global_counts += batch_counts
    
    global_counts = torch.clamp(global_counts, min=1)
    max_count = global_counts.max().float()
    
    penalty_tensor = alpha*(1.0 - (torch.log(global_counts.float()) / torch.log(max_count)))
    return penalty_tensor.unsqueeze(0) # 1, vocab_size


# In[12]:


class RoPECache(nn.Module): 
    def __init__(self, d_k, dropout=0.1, max_len=2048): 
        super().__init__()
        position = torch.arange(0, max_len) # max_len,
        inv_freq = torch.exp(torch.arange(0, d_k, 2) * (-math.log(10000.)) / d_k) # d_k/2 ,
        theta = torch.einsum('i,j->ij', position, inv_freq) # max_len, d_k/2
        emb = torch.cat([theta, theta],dim=-1) # max_len, d_k
        # persistent for not save in .pt
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    def forward(self, seq_len, step=0): 
        return self.cos_cached[step:step+seq_len].unsqueeze(0).unsqueeze(0), self.sin_cached[step:step+seq_len].unsqueeze(0).unsqueeze(0) # 1, 1, seq_len, d_model

def rotate_half(x): 
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin): 
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin

    return q_embed, k_embed

class SwiGLU(nn.Module):
    def __init__(self, d_ff, d_model, dropout=0.1): 
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x): 
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)

class RMSNorm(nn.Module): 
    def __init__(self, d_model, eps=1e-6): 
        super().__init__()
        self.g = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x): 
        x_f32 = x.float()
        rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms.to(x.dtype)) * self.g

class ResidualConnectionWithRMS(nn.Module): 
    def __init__(self, d_model, dropout=0.1): 
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer): 
        return x + self.dropout(sublayer(self.norm(x)))
        

class MultiHeadAttentionWithRoPE(nn.Module): 
    def __init__(self, h, d_model, dropout=0.1): 
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(dropout)
        self.rope_cache = RoPECache(self.d_k)

    def forward(self, q, k, v, mask=None, past_kv=None, use_cache=None, is_cross_attention=False): 
        batch_size = q.size(0)

        if past_kv is not None: 
            if (is_cross_attention): 
                query = self.linears[0](q).view(batch_size, -1, self.h, self.d_k).transpose(1,2)
                key, value = past_kv
            else: 
                query, key, value = [linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1,2) for linear, x in zip(self.linears[:3], (q, k, v))]
                seq_len = query.shape[-2]
                K, V = past_kv 
                step = K.shape[-2]
                cos, sin = self.rope_cache(seq_len, step=step)
                query, key = apply_rotary_pos_emb(query, key, cos, sin)
                key = torch.cat([K, key], dim=-2)
                value = torch.cat([V, value], dim=-2)
        else:
            # batch_size, h, seq_len, d_k
            query, key, value = [linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1,2) for linear, x in zip(self.linears[:3], (q, k, v))]
            if is_cross_attention == False: 
                seq_len = query.shape[-2]
                cos, sin = self.rope_cache(seq_len)
                query, key = apply_rotary_pos_emb(query, key, cos, sin)

        persent_kv = (key, value) if use_cache == True else None
        
        if mask is not None:
            mask = mask.unsqueeze(1)
        x, attn = scaled_dot_product_attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    
        return self.linears[-1](x), persent_kv

class ImprovedEncoderLayer(nn.Module): 
    def __init__(self, d_model, d_ff, h, dropout=0.1): 
        super().__init__()
        self.attn = MultiHeadAttentionWithRoPE(h, d_model)
        self.swiglu = SwiGLU(d_ff,d_model)
        self.residual = clones(ResidualConnectionWithRMS(d_model),2)

    def forward(self, x, mask): 
        x = self.residual[0](x, lambda x: self.attn(x,x,x, mask=mask)[0])
        x = self.residual[1](x, self.swiglu)
        return x

class ImprovedEncoder(nn.Module): 
    def __init__(self, d_model, d_ff, h, N, dropout=0.1): 
        super().__init__()
        self.embedding = nn.Embedding(36000, d_model)
        self.encoder_layers = clones(ImprovedEncoderLayer(d_model,d_ff,h, dropout=dropout), N)
        
    def forward(self, src_ids, mask=None): 
        x = self.embedding(src_ids) 
        for layer in self.encoder_layers:
            x = layer(x,mask)
        return x


class ImprovedDecoderLayer(nn.Module): 
    def __init__(self, d_model, d_ff, h, dropout=0.1): 
        super().__init__()
        self.masked_attn = MultiHeadAttentionWithRoPE(h, d_model)
        self.cross_attn = MultiHeadAttentionWithRoPE(h, d_model)
        self.sublayer = clones(ResidualConnectionWithRMS(d_model), 3)
        self.swiglu = SwiGLU(d_ff, d_model)

    def forward(self, x, memory, src_mask, tgt_mask, past_kv=None, use_cache=False):
        past_self_kv = past_kv[0] if past_kv is not None else None
        past_cross_kv = past_kv[1] if past_kv is not None else None        
        present_self_kv = None
        def self_attn_wrapper(q): 
            nonlocal present_self_kv 
            out, present_self_kv = self.masked_attn(q, q, q, mask=tgt_mask, past_kv=past_self_kv, use_cache=use_cache, is_cross_attention=False)            
            return out
        x = self.sublayer[0](x, self_attn_wrapper)
    
        present_cross_kv = None
        def cross_attn_wrapper(q): 
            nonlocal present_cross_kv
            out, present_cross_kv = self.cross_attn(q, memory, memory, mask=src_mask, past_kv=past_cross_kv, use_cache=use_cache, is_cross_attention=True)            
            return out
        x = self.sublayer[1](x, cross_attn_wrapper)
        x = self.sublayer[2](x, self.swiglu)
        return x, (present_self_kv, present_cross_kv)

class ImprovedDecoder(nn.Module):
  def __init__(self, d_model, d_ff, h, N, dropout=0.1):
    super().__init__()
    self.embedding = nn.Embedding(36000, d_model)
    self.layers = clones(ImprovedDecoderLayer(d_model, d_ff, h, dropout=dropout), N)
    self.linear = nn.Linear(d_model, 36000)
      
  def forward(self, x, memory, src_mask, tgt_mask, past_kvs=None, use_cache=False):
    x = self.embedding(x)
    present_kvs = [] 
    for i, layer in enumerate(self.layers):
      past_kv = past_kvs[i] if past_kvs is not None else None
      x, present_kv = layer(x, memory, src_mask, tgt_mask, past_kv=past_kv, use_cache=use_cache)
      present_kvs.append(present_kv)
    return self.linear(x), present_kvs

class ImprovedBaselineTransformer(nn.Module):
  def __init__(self, d_model, d_ff, h, N, dropout=0.1):
    super().__init__()
    self.encoder = ImprovedEncoder(d_model, d_ff, h, N, dropout=dropout)
    self.decoder = ImprovedDecoder(d_model, d_ff, h, N, dropout=dropout)
    self.d_model = d_model

  def forward(self, src, tgt, src_mask=None, tgt_mask=None):
    memory = self.encoder(src, src_mask)
    output = self.decoder(tgt, memory, src_mask, tgt_mask)[0]
    return output

  @torch.inference_mode()
  def generate_summary(self, src_token, src_mask, bos_idx, eos_idx, pad_idx, max_len=360, strategy='greedy',penalty_tensor=None, base_penalty=0.0, **kwarg):
    device = next(self.parameters()).device
    device_type=device.type
    src_token = src_token.to(device)
    src_mask = src_mask.to(device)
    with torch.autocast(device_type=device_type, dtype=(torch.float16 if device_type == 'cuda' else torch.bfloat16), enabled=(device_type == 'cuda')):
      memory = self.encoder(src_token, src_mask)

    if strategy == 'greedy': 
        return self._greedy_search(src_token, src_mask, memory, bos_idx, eos_idx, pad_idx, max_len)
    elif strategy == 'greedy_with_penalty': 
        return self._greedy_search(src_token, src_mask, memory, bos_idx, eos_idx, pad_idx, max_len, penalty_tensor, base_penalty=base_penalty)

  def _greedy_search(self, src_token, src_mask, memory, bos_idx, eos_idx, pad_idx, max_len, penalty_tensor=None, base_penalty=0.0): 
      device = next(self.parameters()).device
      device_type=device.type
      batch_size = src_token.size(0)
      
      tgt_tokens = torch.full((batch_size,1), bos_idx, dtype=torch.long, device=device)
      unfinished = torch.ones((batch_size,1), dtype=torch.bool, device=device)

      past_kvs = None

      if penalty_tensor is not None:
          # penaldy_tensor shape: 1,vocab_size
          penalty_tensor = penalty_tensor.to(device)
          vocab_size = penalty_tensor.size(1)
          counts = torch.zeros((batch_size, vocab_size), dtype=torch.long, device=device)
      
      for step in range(max_len): 
          with torch.autocast(device_type=device_type, dtype=(torch.float16 if device_type == 'cuda' else torch.bfloat16), enabled=(device_type=='cuda')): 
              input_token = tgt_tokens if step == 0 else tgt_tokens[:, -1:]
              output, present_kvs = self.decoder(input_token, memory, src_mask, None, past_kvs=past_kvs, use_cache=True)
          next_token_logits = output[:, -1, :] # batch_size, vocab_size
          if penalty_tensor is not None: 
              mask = counts > 0
              next_token_logits = next_token_logits - (penalty_tensor * counts + mask * base_penalty)
          next_token = next_token_logits.argmax(dim= -1).unsqueeze(-1) # batch_size, 1
          next_token = next_token * unfinished + (~unfinished) * pad_idx
          tgt_tokens = torch.cat([tgt_tokens, next_token], dim=-1)
          unfinished = unfinished & (eos_idx != next_token)
          past_kvs = present_kvs
          if penalty_tensor is not None: 
              batch_indices = torch.arange(batch_size, device=device)
              active_mask = unfinished.squeeze(-1)
              active_indices = batch_indices[active_mask]
              active_tokens = next_token.squeeze(-1)[active_mask]
              counts[active_indices, active_tokens] += 1
          if unfinished.max() == 0: 
              break
      return tgt_tokens


# In[13]:


class Mamba(nn.Module): pass
class BiMambaBlock(nn.Module): 
    def __init__(self, d_model, d_state=16, d_conv=4, expand=1): 
        super().__init__()
        self.mamba_forward = Mamba(
            d_model=d_model,
            d_state=d_state, # Mặc định tốt nhất
            d_conv=d_conv,   # Mặc định tốt nhất
            expand=expand    # Ép về 1 để công bằng tham số với Transformer Baseline
        )
        self.mamba_backward = Mamba(
            d_model=d_model,
            d_state=d_state, # Mặc định tốt nhất
            d_conv=d_conv,   # Mặc định tốt nhất
            expand=expand    # Ép về 1 để công bằng tham số với Transformer Baseline
        )

    def forward(self, x): 
        # batch_size, seq_len, d_model
        x_forward = self.mamba_forward(x)
        x_flipped = torch.flip(x, dims=[-2]).contiguous()
        x_backward = self.mamba_backward(x_flipped)
        x_backward = torch.flip(x_backward, dims=[-2]).contiguous()
        return x_forward, x_backward

class BiMambaEncoderLayer(nn.Module): 
    def __init__(self, d_model, d_ff_new, dropout=0.1): 
        super().__init__()
        
        self.bi_mamba = BiMambaBlock(d_model=d_model, expand=1) 
        
        self.swiglu = SwiGLU(d_ff_new, d_model)
        
        self.pipe_gate = nn.Linear(d_model, 1)
        self.pipe_semantic = nn.Linear(d_model, d_model)

        self.gate_fusion = nn.Linear(2,1)  
        
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.eps = 1e-6
        

    def forward(self, x, mask, min_p=0.1): 
        # mask: batch_size, 1, seq_len
        mask = mask.to(x.dtype)
        mask = mask.squeeze(-2).unsqueeze(-1) # batch_size, seq_len, 1
        
        x_normed = self.norm(x)
        x_masked = x_normed * mask # batch_size, seq_len, d_model
        x_masked = x_masked + (1.0 - mask) * self.eps # Tránh Nan 
        
        x_forward, x_backward = self.bi_mamba(x_masked)

        x_forward = x_forward * mask
        x_backward = x_backward * mask

        x_forward = self.swiglu(x_forward)
        x_backward = self.swiglu(x_backward)
        
        gate_fwd_logits = self.pipe_gate(x_forward) # batch_size, seq_len, 1
        gate_bwd_logits = self.pipe_gate(x_backward) # batch_size, seq_len, 1

        combined_logits = torch.cat([gate_fwd_logits, gate_bwd_logits], dim=-1) # batch_size, seq_len, 2
        combined_mask = torch.sigmoid(self.gate_fusion(combined_logits)) * mask # batch_size, seq_len, 1  
        
        sem_fwd = self.pipe_semantic(x_forward) * mask 
        sem_bwd = self.pipe_semantic(x_backward) * mask
        
        combined_sem = sem_fwd + sem_bwd

        gated_sem = combined_sem * combined_mask

        final_sem = combined_sem + (gated_sem - gated_sem.detach())

        x = x + self.dropout(final_sem)

        mask_loss = (combined_mask * mask).sum() / (mask.sum() + self.eps)
        
        mask_bool = (combined_mask.squeeze(-1) > min_p) # batch_size, seq_len
        max_k = mask_bool.sum(-1).max().item() 
        max_k = max(1, max_k) 

        topk_scores, topk_indices = torch.topk(combined_mask.squeeze(-1), max_k, dim=1)
        gather_idx = topk_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
        compressed_x = torch.gather(x, dim=1, index=gather_idx) # batch_size, seq_len, d_model
        
        decoder_padding_mask = torch.gather(mask.squeeze(-1), dim=1, index=topk_indices).bool()
        decoder_padding_mask[:, 0] = True
        return compressed_x, decoder_padding_mask , mask_loss

class BiMambaEncoder(nn.Module): 
    def __init__(self, d_model, d_ff,d_ff_new, h, N, dropout=0.1): 
        super().__init__()
        self.embedding = nn.Embedding(36000, d_model)
        self.attn_encode_layer = clones(ImprovedEncoderLayer(d_model,d_ff,h, dropout=dropout), N)
        self.bi_mamba_block = BiMambaEncoderLayer(d_model, d_ff_new, dropout=dropout)

    def forward(self, x, mask=None, min_p=0.1): 
        x = self.embedding(x) # batch_size, seq_len, d_model
        # mask: batch_size, 1, seq_len
        if mask is not None: 
            tmp_mask = mask.squeeze(-2).unsqueeze(-1) # batch_size, seq_len, 1
            x = x * tmp_mask
        for layer in self.attn_encode_layer: 
            x = layer(x,mask)
        output, new_mask, mask_loss = self.bi_mamba_block(x,mask, min_p = min_p)
                                                          
        new_mask = new_mask.unsqueeze(-2)  # (batch, seq_len, 1) -> (batch, 1, seq_len)
        return output, new_mask, mask_loss

class HyperSphereLinear(nn.Module): 
    def __init__(self, d_model, vocab_size, initial_gamma=10.0, tied_weight=None): 
        super().__init__()
        if tied_weight is not None:
            self.weight = tied_weight
        else:
            self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
            nn.init.xavier_uniform_(self.weight)
        self.gamma = nn.Parameter(torch.tensor(initial_gamma))
        self.eps = 1e-8
        self.register_buffer('output_scale', torch.tensor(1.0))

    def forward(self, x): 
        x_norm = F.normalize(x, p=2, dim=-1, eps=self.eps)
        w_norm = F.normalize(self.weight, p=2, dim=-1, eps=self.eps)
        gamma_clamped = torch.clamp(self.gamma, min=0.1, max=50.0)
        cosine_sim = F.linear(x_norm, w_norm)  
        logits = cosine_sim * gamma_clamped
        logits = torch.clamp(logits, min=-11.0, max=11.0)
        return logits



class HyperSphereTransformerDecoder(nn.Module): 
  def __init__(self, d_model, d_ff, h, N, dropout=0.1, tied_weight=None):
    super().__init__()
    self.embedding = nn.Embedding(36000, d_model)
    self.layers = clones(ImprovedDecoderLayer(d_model, d_ff, h, dropout=dropout), N)
    self.linear = HyperSphereLinear(d_model, 36000, 
                                     tied_weight=tied_weight)     
      
  def forward(self, x, memory, src_mask, tgt_mask, past_kvs=None, use_cache=False):
    x = self.embedding(x)
    present_kvs = [] 
    for i, layer in enumerate(self.layers):
      past_kv = past_kvs[i] if past_kvs is not None else None
      x, present_kv = layer(x, memory, src_mask, tgt_mask, past_kv=past_kv, use_cache=use_cache)
      present_kvs.append(present_kv)
    return self.linear(x), present_kvs


class TransformerWithSoftPromptMamba(nn.Module):
  def __init__(self, d_model, d_ff, d_ff_new, h, N, dropout=0.1):
    super().__init__()
    self.encoder = BiMambaEncoder(d_model, d_ff,d_ff_new, h, N-1, dropout=dropout)
    self.decoder = HyperSphereTransformerDecoder(d_model, d_ff, h, N, dropout=dropout)
    self.d_model = d_model

  def forward(self, src, tgt, src_mask=None, tgt_mask=None):
    memory, src_mask, mask_loss = self.encoder(src, src_mask)
    output = self.decoder(tgt, memory, src_mask, tgt_mask)[0]
    return output, mask_loss

  @torch.inference_mode()
  def generate_summary(self, src_token, src_mask, bos_idx, eos_idx, pad_idx, max_len=360, strategy='greedy',penalty_tensor=None, base_penalty=0.0, **kwarg):
    device = next(self.parameters()).device
    device_type=device.type
    src_token = src_token.to(device)
    src_mask = src_mask.to(device)
    with torch.autocast(device_type=device_type, dtype=(torch.float16 if device_type == 'cuda' else torch.bfloat16), enabled=(device_type == 'cuda')):
      memory, src_mask, _ = self.encoder(src_token, src_mask)

    if strategy == 'greedy': 
        return self._greedy_search(src_token, src_mask, memory, bos_idx, eos_idx, pad_idx, max_len)
    elif strategy == 'greedy_with_penalty': 
        return self._greedy_search(src_token, src_mask, memory, bos_idx, eos_idx, pad_idx, max_len, penalty_tensor, base_penalty=base_penalty)

  def _greedy_search(self, src_token, src_mask, memory, bos_idx, eos_idx, pad_idx, max_len, penalty_tensor=None, base_penalty=0.0): 
      device = next(self.parameters()).device
      device_type=device.type
      batch_size = src_token.size(0)
      
      tgt_tokens = torch.full((batch_size,1), bos_idx, dtype=torch.long, device=device)
      unfinished = torch.ones((batch_size,1), dtype=torch.bool, device=device)

      past_kvs = None

      if penalty_tensor is not None:
          # penaldy_tensor shape: 1,vocab_size
          penalty_tensor = penalty_tensor.to(device)
          vocab_size = penalty_tensor.size(1)
          counts = torch.zeros((batch_size, vocab_size), dtype=torch.long, device=device)
      
      for step in range(max_len): 
          with torch.autocast(device_type=device_type, dtype=(torch.float16 if device_type == 'cuda' else torch.bfloat16), enabled=(device_type=='cuda')): 
              input_token = tgt_tokens if step == 0 else tgt_tokens[:, -1:]
              output, present_kvs = self.decoder(input_token, memory, src_mask, None, past_kvs=past_kvs, use_cache=True)
          next_token_logits = output[:, -1, :] # batch_size, vocab_size
          if penalty_tensor is not None: 
              mask = counts > 0
              next_token_logits = next_token_logits - (penalty_tensor * counts + mask * base_penalty)
          next_token = next_token_logits.argmax(dim= -1).unsqueeze(-1) # batch_size, 1
          next_token = next_token * unfinished + (~unfinished) * pad_idx
          tgt_tokens = torch.cat([tgt_tokens, next_token], dim=-1)
          unfinished = unfinished & (eos_idx != next_token)
          past_kvs = present_kvs
          if penalty_tensor is not None: 
              batch_indices = torch.arange(batch_size, device=device)
              active_mask = unfinished.squeeze(-1)
              active_indices = batch_indices[active_mask]
              active_tokens = next_token.squeeze(-1)[active_mask]
              counts[active_indices, active_tokens] += 1
          if unfinished.max() == 0: 
              break
      return tgt_tokens


# In[14]:


# 1. LAYER MỚI: Kế thừa lại toàn bộ các tầng tuyến tính của Layer cũ
class EntityGatedBiMambaEncoderLayer(BiMambaEncoderLayer): 
    def forward(self, x, mask, src_ner_mask=None, min_p=0.1): 
        # mask: batch_size, 1, seq_len
        mask = mask.to(x.dtype)
        mask = mask.squeeze(-2).unsqueeze(-1) # batch_size, seq_len, 1
        
        x_normed = self.norm(x)
        x_masked = x_normed * mask 
        x_masked = x_masked + (1.0 - mask) * self.eps 
        
        x_forward, x_backward = self.bi_mamba(x_masked)

        x_forward = x_forward * mask
        x_backward = x_backward * mask

        x_forward = self.swiglu(x_forward)
        x_backward = self.swiglu(x_backward)
        
        gate_fwd_logits = self.pipe_gate(x_forward) 
        gate_bwd_logits = self.pipe_gate(x_backward) 

        combined_logits = torch.cat([gate_fwd_logits, gate_bwd_logits], dim=-1) 
        combined_mask = torch.sigmoid(self.gate_fusion(combined_logits)) * mask   
        
        sem_fwd = self.pipe_semantic(x_forward) * mask 
        sem_bwd = self.pipe_semantic(x_backward) * mask
        
        combined_sem = sem_fwd + sem_bwd
        gated_sem = combined_sem * combined_mask
        final_sem = combined_sem + (gated_sem - gated_sem.detach())

        x = x + self.dropout(final_sem)

        base_mask_loss = (combined_mask * mask).sum() / (mask.sum() + self.eps)
        
        if src_ner_mask is not None and src_ner_mask.sum() > 0:
            target_ner = src_ner_mask.unsqueeze(-1).to(x.dtype)
            combined_mask_clamped = combined_mask.clamp(1e-6, 1 - 1e-6)
            ner_loss = F.binary_cross_entropy(combined_mask_clamped, target_ner, reduction='none')
            actual_ner_loss = (ner_loss * target_ner).sum() / (target_ner.sum() + self.eps)
            
            total_mask_loss = base_mask_loss + actual_ner_loss
        else:
            total_mask_loss = base_mask_loss

        if src_ner_mask is not None and src_ner_mask.sum() > 0:
            ner_bool = (src_ner_mask > 0).unsqueeze(-1).bool()
            combined_mask = torch.where(ner_bool, torch.ones_like(combined_mask), combined_mask)
            
        mask_bool = (combined_mask.squeeze(-1) > min_p) 
        max_k = mask_bool.sum(-1).max().item() 
        max_k = max(1, max_k) 

        topk_scores, topk_indices = torch.topk(combined_mask.squeeze(-1), max_k, dim=1)
        gather_idx = topk_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
        compressed_x = torch.gather(x, dim=1, index=gather_idx) 
        
        decoder_padding_mask = torch.gather(mask.squeeze(-1), dim=1, index=topk_indices).bool()
        decoder_padding_mask[:, 0] = True
        
        return compressed_x, decoder_padding_mask, total_mask_loss

class EntityGatedBiMambaEncoder(BiMambaEncoder):
    def __init__(self, d_model, d_ff, d_ff_new, h, N, dropout=0.1):
        super().__init__(d_model, d_ff, d_ff_new, h, N, dropout=dropout)
        self.bi_mamba_block = EntityGatedBiMambaEncoderLayer(d_model, d_ff_new, dropout=dropout)

    def forward(self, x, mask=None, src_ner_mask=None, min_p=0.1): 
        x = self.embedding(x) 
        if mask is not None: 
            tmp_mask = mask.squeeze(-2).unsqueeze(-1) 
            x = x * tmp_mask
        for layer in self.attn_encode_layer: 
            x = layer(x, mask)
            
        output, new_mask, mask_loss = self.bi_mamba_block(x, mask, src_ner_mask=src_ner_mask, min_p=min_p)
                                                                    
        new_mask = new_mask.unsqueeze(-2)  
        return output, new_mask, mask_loss

class EntityGatedMambaSeq2Seq(TransformerWithSoftPromptMamba):
    def __init__(self, d_model, d_ff, d_ff_new, h, N, dropout=0.1):
        super().__init__(d_model, d_ff, d_ff_new, h, N, dropout=dropout)
        self.encoder = EntityGatedBiMambaEncoder(d_model, d_ff, d_ff_new, h, N-1, dropout=dropout)
        self.decoder = HyperSphereTransformerDecoder(
            d_model, d_ff, h, N, 
            dropout=dropout,
            tied_weight=self.encoder.embedding.weight)
        self.decoder.embedding.weight = self.encoder.embedding.weight
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_ner_mask=None):
        memory, src_mask, mask_loss = self.encoder(src, src_mask, src_ner_mask=src_ner_mask)
        output = self.decoder(tgt, memory, src_mask, tgt_mask)[0]
        
        return output, mask_loss
        
    @torch.inference_mode()
    def generate_summary(self, src_token, src_mask, bos_idx, eos_idx, pad_idx, max_len=360, strategy='greedy',penalty_tensor=None, base_penalty=0.0, **kwarg):
        device = next(self.parameters()).device
        device_type=device.type
        src_token = src_token.to(device)
        src_mask = src_mask.to(device)
        with torch.autocast(device_type=device_type, dtype=(torch.float16 if device_type == 'cuda' else torch.bfloat16), enabled=(device_type == 'cuda')):
          memory, src_mask, _ = self.encoder(src_token, src_mask)
    
        if strategy == 'greedy': 
            return self._greedy_search(src_token, src_mask, memory, bos_idx, eos_idx, pad_idx, max_len)
        elif strategy == 'greedy_with_penalty': 
            return self._greedy_search(src_token, src_mask, memory, bos_idx, eos_idx, pad_idx, max_len, penalty_tensor, base_penalty=base_penalty)

    def _greedy_search(self, src_token, src_mask, memory, bos_idx, eos_idx, pad_idx, max_len, penalty_tensor=None, base_penalty=0.0): 
        device = next(self.parameters()).device
        device_type=device.type
        batch_size = src_token.size(0)
        
        tgt_tokens = torch.full((batch_size,1), bos_idx, dtype=torch.long, device=device)
        unfinished = torch.ones((batch_size,1), dtype=torch.bool, device=device)
        
        past_kvs = None
        
        if penalty_tensor is not None:
          # penaldy_tensor shape: 1,vocab_size
          penalty_tensor = penalty_tensor.to(device)
          vocab_size = penalty_tensor.size(1)
          counts = torch.zeros((batch_size, vocab_size), dtype=torch.long, device=device)
        
        for step in range(max_len): 
          with torch.autocast(device_type=device_type, dtype=(torch.float16 if device_type == 'cuda' else torch.bfloat16), enabled=(device_type=='cuda')): 
              input_token = tgt_tokens if step == 0 else tgt_tokens[:, -1:]
              output, present_kvs = self.decoder(input_token, memory, src_mask, None, past_kvs=past_kvs, use_cache=True)
          next_token_logits = output[:, -1, :] # batch_size, vocab_size
          if penalty_tensor is not None: 
              mask = counts > 0
              next_token_logits = next_token_logits - (penalty_tensor * counts + mask * base_penalty)
          next_token = next_token_logits.argmax(dim= -1).unsqueeze(-1) # batch_size, 1
          next_token = next_token * unfinished + (~unfinished) * pad_idx
          tgt_tokens = torch.cat([tgt_tokens, next_token], dim=-1)
          unfinished = unfinished & (eos_idx != next_token)
          past_kvs = present_kvs
          if penalty_tensor is not None: 
              batch_indices = torch.arange(batch_size, device=device)
              active_mask = unfinished.squeeze(-1)
              active_indices = batch_indices[active_mask]
              active_tokens = next_token.squeeze(-1)[active_mask]
              counts[active_indices, active_tokens] += 1
          if unfinished.max() == 0: 
              break
        return tgt_tokens
    


# In[15]:


class TransformerWithWeightTyingAndHyperSphereNorm(nn.Module): 
  def __init__(self, d_model, d_ff, h, N, dropout=0.1):
    super().__init__()
    self.encoder = ImprovedEncoder(d_model, d_ff, h, N, dropout=dropout)
    self.decoder = HyperSphereTransformerDecoder(
            d_model, d_ff, h, N, 
            dropout=dropout,
            tied_weight=self.encoder.embedding.weight)
    self.decoder.embedding.weight = self.encoder.embedding.weight
    self.d_model = d_model

  def forward(self, src, tgt, src_mask=None, tgt_mask=None):
    memory = self.encoder(src, src_mask)
    output = self.decoder(tgt, memory, src_mask, tgt_mask)[0]
    return output

  @torch.inference_mode()
  def generate_summary(self, src_token, src_mask, bos_idx, eos_idx, pad_idx, max_len=360, strategy='greedy',penalty_tensor=None, base_penalty=0.0, **kwarg):
    device = next(self.parameters()).device
    device_type=device.type
    src_token = src_token.to(device)
    src_mask = src_mask.to(device)
    with torch.autocast(device_type=device_type, dtype=(torch.float16 if device_type == 'cuda' else torch.bfloat16), enabled=(device_type == 'cuda')):
      memory = self.encoder(src_token, src_mask)

    if strategy == 'greedy': 
        return self._greedy_search(src_token, src_mask, memory, bos_idx, eos_idx, pad_idx, max_len)
    elif strategy == 'greedy_with_penalty': 
        return self._greedy_search(src_token, src_mask, memory, bos_idx, eos_idx, pad_idx, max_len, penalty_tensor, base_penalty=base_penalty)

  def _greedy_search(self, src_token, src_mask, memory, bos_idx, eos_idx, pad_idx, max_len, penalty_tensor=None, base_penalty=0.0): 
      device = next(self.parameters()).device
      device_type=device.type
      batch_size = src_token.size(0)
      
      tgt_tokens = torch.full((batch_size,1), bos_idx, dtype=torch.long, device=device)
      unfinished = torch.ones((batch_size,1), dtype=torch.bool, device=device)

      past_kvs = None

      if penalty_tensor is not None:
          # penaldy_tensor shape: 1,vocab_size
          penalty_tensor = penalty_tensor.to(device)
          vocab_size = penalty_tensor.size(1)
          counts = torch.zeros((batch_size, vocab_size), dtype=torch.long, device=device)
      
      for step in range(max_len): 
          with torch.autocast(device_type=device_type, dtype=(torch.float16 if device_type == 'cuda' else torch.bfloat16), enabled=(device_type=='cuda')): 
              input_token = tgt_tokens if step == 0 else tgt_tokens[:, -1:]
              output, present_kvs = self.decoder(input_token, memory, src_mask, None, past_kvs=past_kvs, use_cache=True)
          next_token_logits = output[:, -1, :] # batch_size, vocab_size
          if penalty_tensor is not None: 
              mask = counts > 0
              next_token_logits = next_token_logits - (penalty_tensor * counts + mask * base_penalty)
          next_token = next_token_logits.argmax(dim= -1).unsqueeze(-1) # batch_size, 1
          next_token = next_token * unfinished + (~unfinished) * pad_idx
          tgt_tokens = torch.cat([tgt_tokens, next_token], dim=-1)
          unfinished = unfinished & (eos_idx != next_token)
          past_kvs = present_kvs
          if penalty_tensor is not None: 
              batch_indices = torch.arange(batch_size, device=device)
              active_mask = unfinished.squeeze(-1)
              active_indices = batch_indices[active_mask]
              active_tokens = next_token.squeeze(-1)[active_mask]
              counts[active_indices, active_tokens] += 1
          if unfinished.max() == 0: 
              break
      return tgt_tokens


# In[16]:


class EntityGuidedBiMambaEncoderLayer(nn.Module): 
    def __init__(self, d_model, d_ff_new, dropout=0.1): 
        super().__init__()
        
        self.bi_mamba = BiMambaBlock(d_model=d_model, expand=1) 
        self.swiglu = SwiGLU(d_ff_new, d_model)
        
        self.entity_predictor = nn.Linear(d_model, 1)
        self.pipe_semantic = nn.Linear(d_model, d_model)
        
        self.prob_to_prompt = nn.Linear(1, d_model, bias=False)
        
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.eps = 1e-6
        
    def forward(self, x, mask, src_ner_mask=None, min_p=0.3): 
        # mask: batch_size, 1, seq_len
        mask = mask.to(x.dtype)
        mask = mask.squeeze(-2).unsqueeze(-1) # batch_size, seq_len, 1
        
        combined_logits = self.entity_predictor(x)
        combined_mask = torch.sigmoid(combined_logits) * mask
        
        mask_loss = torch.tensor(0.0, device=x.device)
        if src_ner_mask is not None and src_ner_mask.sum() > 0:
            target_ner = src_ner_mask.unsqueeze(-1).to(x.dtype)
            combined_mask_clamped = combined_mask.clamp(1e-6, 1 - 1e-6)
            mask_loss = F.binary_cross_entropy(combined_mask_clamped, target_ner, reduction='mean')
            
            ner_bool = (src_ner_mask > 0).unsqueeze(-1).bool()
            combined_mask = torch.where(ner_bool, torch.ones_like(combined_mask), combined_mask)
            
        prompt_vector = self.prob_to_prompt(combined_mask)
        x_injected = x + prompt_vector
        
        x_normed = self.norm(x_injected)
        x_masked = x_normed * mask 
        x_masked = x_masked + (1.0 - mask) * self.eps 
        
        x_forward, x_backward = self.bi_mamba(x_masked)

        x_forward = x_forward * mask
        x_backward = x_backward * mask

        x_forward = self.swiglu(x_forward)
        x_backward = self.swiglu(x_backward)
        
        sem_fwd = self.pipe_semantic(x_forward) * mask 
        sem_bwd = self.pipe_semantic(x_backward) * mask
        combined_sem = sem_fwd + sem_bwd

        output = x_injected + self.dropout(combined_sem)
        
        return output, mask_loss

class EntityGuidedBiMambaEncoder(nn.Module): 
    def __init__(self, d_model, d_ff, d_ff_new, h, N, dropout=0.1): 
        super().__init__()
        self.embedding = nn.Embedding(36000, d_model)
        self.attn_encode_layer = clones(ImprovedEncoderLayer(d_model,d_ff,h, dropout=dropout), N)
        self.bi_mamba_block = EntityGuidedBiMambaEncoderLayer(d_model, d_ff_new, dropout=dropout)

    def forward(self, x, mask=None, src_ner_mask=None, min_p=0.3): 
        x = self.embedding(x) 
        if mask is not None: 
            tmp_mask = mask.squeeze(-2).unsqueeze(-1) 
            x = x * tmp_mask
        for layer in self.attn_encode_layer: 
            x = layer(x, mask)
            
        output, mask_loss = self.bi_mamba_block(x, mask, src_ner_mask=src_ner_mask, min_p=min_p)
                                          
        return output, mask_loss
        
class EntityGuidedHybridMamba(nn.Module): 
    def __init__(self, d_model, d_ff, d_ff_new, h, N, dropout=0.1):
        super().__init__()
        self.encoder = EntityGuidedBiMambaEncoder(d_model, d_ff, d_ff_new, h, N-1, dropout=dropout)
        self.decoder = HyperSphereTransformerDecoder(
            d_model, d_ff, h, N, 
            dropout=dropout,
            tied_weight=self.encoder.embedding.weight)
        self.decoder.embedding.weight = self.encoder.embedding.weight
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_ner_mask=None):
        memory, mask_loss = self.encoder(src, src_mask, src_ner_mask=src_ner_mask)
        output = self.decoder(tgt, memory, src_mask, tgt_mask)[0]
        
        return output, mask_loss
        
    @torch.inference_mode()
    def generate_summary(self, src_token, src_mask, bos_idx, eos_idx, pad_idx, max_len=360, strategy='greedy',penalty_tensor=None, base_penalty=0.0, **kwarg):
        device = next(self.parameters()).device
        device_type=device.type
        src_token = src_token.to(device)
        src_mask = src_mask.to(device)
        with torch.autocast(device_type=device_type, dtype=(torch.float16 if device_type == 'cuda' else torch.bfloat16), enabled=(device_type == 'cuda')):
          memory, _ = self.encoder(src_token, src_mask)
    
        if strategy == 'greedy': 
            return self._greedy_search(src_token, src_mask, memory, bos_idx, eos_idx, pad_idx, max_len)
        elif strategy == 'greedy_with_penalty': 
            return self._greedy_search(src_token, src_mask, memory, bos_idx, eos_idx, pad_idx, max_len, penalty_tensor, base_penalty=base_penalty)

    def _greedy_search(self, src_token, src_mask, memory, bos_idx, eos_idx, pad_idx, max_len, penalty_tensor=None, base_penalty=0.0): 
        device = next(self.parameters()).device
        device_type=device.type
        batch_size = src_token.size(0)
        
        tgt_tokens = torch.full((batch_size,1), bos_idx, dtype=torch.long, device=device)
        unfinished = torch.ones((batch_size,1), dtype=torch.bool, device=device)
        
        past_kvs = None
        
        if penalty_tensor is not None:
          # penaldy_tensor shape: 1,vocab_size
          penalty_tensor = penalty_tensor.to(device)
          vocab_size = penalty_tensor.size(1)
          counts = torch.zeros((batch_size, vocab_size), dtype=torch.long, device=device)
        
        for step in range(max_len): 
          with torch.autocast(device_type=device_type, dtype=(torch.float16 if device_type == 'cuda' else torch.bfloat16), enabled=(device_type=='cuda')): 
              input_token = tgt_tokens if step == 0 else tgt_tokens[:, -1:]
              output, present_kvs = self.decoder(input_token, memory, src_mask, None, past_kvs=past_kvs, use_cache=True)
          next_token_logits = output[:, -1, :] # batch_size, vocab_size
          if penalty_tensor is not None: 
              mask = counts > 0
              next_token_logits = next_token_logits - (penalty_tensor * counts + mask * base_penalty)
          next_token = next_token_logits.argmax(dim= -1).unsqueeze(-1) # batch_size, 1
          next_token = next_token * unfinished + (~unfinished) * pad_idx
          tgt_tokens = torch.cat([tgt_tokens, next_token], dim=-1)
          unfinished = unfinished & (eos_idx != next_token)
          past_kvs = present_kvs
          if penalty_tensor is not None: 
              batch_indices = torch.arange(batch_size, device=device)
              active_mask = unfinished.squeeze(-1)
              active_indices = batch_indices[active_mask]
              active_tokens = next_token.squeeze(-1)[active_mask]
              counts[active_indices, active_tokens] += 1
          if unfinished.max() == 0: 
              break
        return tgt_tokens


# In[17]:


class EntityGuidedTransformerEncoderLayer(nn.Module): 
    def __init__(self, d_model, d_ff, h, dropout=0.1): 
        super().__init__()
        self.attn = MultiHeadAttentionWithRoPE(h, d_model)
        self.swiglu = SwiGLU(d_ff, d_model)
        self.residual = clones(ResidualConnectionWithRMS(d_model), 2)
        
        self.entity_predictor = nn.Linear(d_model, 1)
        self.prob_to_prompt = nn.Linear(1, d_model, bias=False)
        
    def forward(self, x, mask, src_ner_mask=None):
        combined_logits = self.entity_predictor(x)
        combined_mask = torch.sigmoid(combined_logits) * mask.to(x.dtype).squeeze(-2).unsqueeze(-1)
        
        mask_loss = torch.tensor(0.0, device=x.device)
        if src_ner_mask is not None and src_ner_mask.sum() > 0:
            target_ner = src_ner_mask.unsqueeze(-1).to(x.dtype)
            combined_mask_clamped = combined_mask.clamp(1e-6, 1 - 1e-6)
            mask_loss = F.binary_cross_entropy(combined_mask_clamped, target_ner, reduction='mean')
            
            ner_bool = (src_ner_mask > 0).unsqueeze(-1).bool()
            combined_mask = torch.where(ner_bool, torch.ones_like(combined_mask), combined_mask)
            
        prompt_vector = self.prob_to_prompt(combined_mask)
        x_injected = x + prompt_vector
        
        x = self.residual[0](x_injected, lambda x: self.attn(x, x, x, mask=mask)[0])
        x = self.residual[1](x, self.swiglu)
        
        return x, mask_loss



class EntityGuidedPureTransformerEncoder(nn.Module): 
    def __init__(self, d_model, d_ff, d_ff_new, h, N, dropout=0.1): 
        super().__init__()
        self.embedding = nn.Embedding(36000, d_model)
        self.attn_encode_layer = clones(ImprovedEncoderLayer(d_model,d_ff,h, dropout=dropout), N)
        self.entity_guided_block = EntityGuidedTransformerEncoderLayer(d_model, d_ff_new,h, dropout=dropout)

    def forward(self, x, mask=None, src_ner_mask=None): 
        x = self.embedding(x) 
        if mask is not None: 
            tmp_mask = mask.squeeze(-2).unsqueeze(-1) 
            x = x * tmp_mask
        for layer in self.attn_encode_layer: 
            x = layer(x, mask)
            
        output, mask_loss = self.entity_guided_block(x, mask, src_ner_mask=src_ner_mask)
                                          
        return output, mask_loss
        
class EntityGuidedPureTransformer(nn.Module): 
    def __init__(self, d_model, d_ff, d_ff_new, h, N, dropout=0.1):
        super().__init__()
        self.encoder = EntityGuidedPureTransformerEncoder(d_model, d_ff, d_ff_new, h, N-1, dropout=dropout)
        self.decoder = HyperSphereTransformerDecoder(
            d_model, d_ff, h, N, 
            dropout=dropout,
            tied_weight=self.encoder.embedding.weight)
        self.decoder.embedding.weight = self.encoder.embedding.weight
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_ner_mask=None):
        memory, mask_loss = self.encoder(src, src_mask, src_ner_mask=src_ner_mask)
        output = self.decoder(tgt, memory, src_mask, tgt_mask)[0]
        
        return output, mask_loss
        
    @torch.inference_mode()
    def generate_summary(self, src_token, src_mask, bos_idx, eos_idx, pad_idx, max_len=360, strategy='greedy',penalty_tensor=None, base_penalty=0.0, **kwarg):
        device = next(self.parameters()).device
        device_type=device.type
        src_token = src_token.to(device)
        src_mask = src_mask.to(device)
        with torch.autocast(device_type=device_type, dtype=(torch.float16 if device_type == 'cuda' else torch.bfloat16), enabled=(device_type == 'cuda')):
          memory, _ = self.encoder(src_token, src_mask)
    
        if strategy == 'greedy': 
            return self._greedy_search(src_token, src_mask, memory, bos_idx, eos_idx, pad_idx, max_len)
        elif strategy == 'greedy_with_penalty': 
            return self._greedy_search(src_token, src_mask, memory, bos_idx, eos_idx, pad_idx, max_len, penalty_tensor, base_penalty=base_penalty)

    def _greedy_search(self, src_token, src_mask, memory, bos_idx, eos_idx, pad_idx, max_len, penalty_tensor=None, base_penalty=0.0): 
        device = next(self.parameters()).device
        device_type=device.type
        batch_size = src_token.size(0)
        
        tgt_tokens = torch.full((batch_size,1), bos_idx, dtype=torch.long, device=device)
        unfinished = torch.ones((batch_size,1), dtype=torch.bool, device=device)
        
        past_kvs = None
        
        if penalty_tensor is not None:
          # penaldy_tensor shape: 1,vocab_size
          penalty_tensor = penalty_tensor.to(device)
          vocab_size = penalty_tensor.size(1)
          counts = torch.zeros((batch_size, vocab_size), dtype=torch.long, device=device)
        
        for step in range(max_len): 
          with torch.autocast(device_type=device_type, dtype=(torch.float16 if device_type == 'cuda' else torch.bfloat16), enabled=(device_type=='cuda')): 
              input_token = tgt_tokens if step == 0 else tgt_tokens[:, -1:]
              output, present_kvs = self.decoder(input_token, memory, src_mask, None, past_kvs=past_kvs, use_cache=True)
          next_token_logits = output[:, -1, :] # batch_size, vocab_size
          if penalty_tensor is not None: 
              mask = counts > 0
              next_token_logits = next_token_logits - (penalty_tensor * counts + mask * base_penalty)
          next_token = next_token_logits.argmax(dim= -1).unsqueeze(-1) # batch_size, 1
          next_token = next_token * unfinished + (~unfinished) * pad_idx
          tgt_tokens = torch.cat([tgt_tokens, next_token], dim=-1)
          unfinished = unfinished & (eos_idx != next_token)
          past_kvs = present_kvs
          if penalty_tensor is not None: 
              batch_indices = torch.arange(batch_size, device=device)
              active_mask = unfinished.squeeze(-1)
              active_indices = batch_indices[active_mask]
              active_tokens = next_token.squeeze(-1)[active_mask]
              counts[active_indices, active_tokens] += 1
          if unfinished.max() == 0: 
              break
        return tgt_tokens


# In[18]:


def subsequence_mask(size):
  attn_shape = (1, size, size)
  return torch.tril(torch.ones(attn_shape).type(torch.bool))

