import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Transformer(nn.Module):
    def __init__(self, src_vocab_size=16000, tgt_vocab_size=16000, d_model=512, n_heads=8, n_layers=6, dropout=0.1, d_ff=2048):
        super().__init__()
        self.d_model = d_model
        
        # SHARED EMBEDDING SCALING
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # PRE-NORM ARCHITECTURE
        self.encoder = nn.ModuleList([EncoderLayer(d_model, n_heads, dropout, d_ff) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, n_heads, dropout, d_ff) for _ in range(n_layers)])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt_input, src_mask=None, tgt_mask=None):
        # Embed and scale
        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt_input) * math.sqrt(self.d_model))

        enc_out = src_emb
        for layer in self.encoder:
            enc_out = layer(enc_out, src_mask)
        
        dec_out = tgt_emb
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out, tgt_mask, src_mask)

        return self.output_layer(self.final_norm(dec_out))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # FUSED QKV PROJECTION 
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, mask=None, kv_source=None):
        batch_size, seq_len, d_model = x.shape
        
        # Self-attention or Cross-attention logic
        if kv_source is None:
            # Self-attention: Project x into Q, K, V at once
            qkv = self.qkv_proj(x).chunk(3, dim=-1)
            q, k, v = [t.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) for t in qkv]
        else:
            # Cross-attention: Q comes from x, K/V from encoder output
            q = F.linear(x, self.qkv_proj.weight[:d_model], self.qkv_proj.bias[:d_model])
            kv = F.linear(kv_source, self.qkv_proj.weight[d_model:], self.qkv_proj.bias[d_model:])
            k, v = kv.chunk(2, dim=-1)
            
            q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask, 
            dropout_p=self.dropout if self.training else 0,
            is_causal=False
        )
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(attn_out)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(), # more efficient/stable than ReLU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Norm applied BEFORE the sub-layer
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        x = x + self.dropout(self.self_attn(self.norm1(x), tgt_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), src_mask, kv_source=enc_out))
        x = x + self.dropout(self.ff(self.norm3(x)))
        return x



def create_padding_mask(seq, pad_idx=0):
    """
    Creates mask for padding tokens. 
    Returns: [Batch, 1, 1, Seq_Len]
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_causal_mask(seq_len):
    """
    Creates triangular mask to prevent attending to future tokens.
    Returns: [1, 1, Seq_Len, Seq_Len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return ~mask.unsqueeze(0).unsqueeze(0)
