from typing import Tuple
import math
import torch
import torch.nn as nn


class SelfAttentionBlock(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 key_query_dim: int,
                 num_heads: int,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.key_query_dim = key_query_dim
        self.num_heads = num_heads

        if (hidden_dim % num_heads != 0) or (key_query_dim % num_heads != 0):
            raise ValueError("Hidden dim not multiple of num heads.")
        self.head_dim = key_query_dim // num_heads

        self.norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.query_proj = nn.Linear(hidden_dim, key_query_dim)
        self.key_proj = nn.Linear(hidden_dim, key_query_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_scale = math.sqrt(1.0 / self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop_prob, inplace=False)
        self.drop = nn.Dropout(drop_prob, inplace=True)

    def _reshape_qkv(self, x: torch.Tensor) -> torch.Tensor:
        # (b, s, d) -> (b, s, h, d/h) -> (b, h, s, d/h)
        b, s, d = x.shape
        x = x.view(b, s, self.num_heads, -1).transpose(1, 2).contiguous()
        return x

    def _reshape_output(self, x: torch.Tensor) -> torch.Tensor:  # noqa
        # (b, h, s, d/h) -> (b, s, h, d/h) -> (b, s, d)
        b, h, s, _ = x.shape
        x = x.transpose(1, 2).reshape(b, s, -1)
        return x

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Self-attention sub-block in Transformer
        
        :param hidden:      (batch_size, num_patches, hidden_dim)
        :return:            (batch_size, num_patches, hidden_dim)
                            (batch_size, num_heads, num_patches, num_patches)
        """
        batch_size, num_patches, _ = hidden.shape

        residual = hidden

        # projection
        hidden = self.norm(hidden)
        q = self.query_proj(hidden)  # (b, s, d)
        k = self.key_proj(hidden)  # (b, s, d)
        v = self.value_proj(hidden)  # (b, s, d)

        # reshape
        q = self._reshape_qkv(q)  # (b, h, s, d/h)
        k = self._reshape_qkv(k)  # (b, h, s, d/h)
        v = self._reshape_qkv(v)  # (b, h, s, d/h)

        # scaled dot-product
        # (b, h, s, d/h) x (b, h, d/h, s) = (b, h, s, s)
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn *= self.attn_scale
        attn = torch.softmax(attn, dim=-1)  # (b, h, s, s)
        attn_drop = self.attn_drop(attn)

        # value multiply
        # (b, h, s, s) x (b, h, s, d/h) = (b, h, s, d/h)
        output = torch.matmul(attn_drop, v)
        output = self._reshape_output(output)  # (b, s, d)

        # output projection
        output = self.out_proj(output)
        output = self.drop(output)

        output = output + residual

        return output, attn


class PostNormSelfAttentionBlock(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 key_query_dim: int,
                 num_heads: int,
                 attn_drop_prob: float = 0.0,
                 drop_prob: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.key_query_dim = key_query_dim
        self.num_heads = num_heads

        if (hidden_dim % num_heads != 0) or ( key_query_dim % num_heads != 0):
            raise ValueError("Hidden dim not multiple of num heads.")
        self.head_dim = key_query_dim // num_heads

        self.query_proj = nn.Linear(hidden_dim, key_query_dim)
        self.key_proj = nn.Linear(hidden_dim, key_query_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_scale = math.sqrt(1.0 / self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop_prob, inplace=False)
        self.drop = nn.Dropout(drop_prob, inplace=True)

        self.norm = nn.LayerNorm(hidden_dim, eps=1e-5)

    def _reshape_qkv(self, x: torch.Tensor) -> torch.Tensor:
        # (b, s, d) -> (b, s, h, d/h) -> (b, h, s, d/h)
        b, s, d = x.shape
        x = x.view(b, s, self.num_heads, -1).transpose(1, 2).contiguous()
        return x

    def _reshape_output(self, x: torch.Tensor) -> torch.Tensor:  # noqa
        # (b, h, s, d/h) -> (b, s, h, d/h) -> (b, s, d)
        b, h, s, _ = x.shape
        x = x.transpose(1, 2).reshape(b, s, -1)
        return x

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Self-attention sub-block in Transformer

        :param hidden:      (batch_size, num_patches, hidden_dim)
        :return:            (batch_size, num_patches, hidden_dim)
                            (batch_size, num_heads, num_patches, num_patches)
        """
        batch_size, num_patches, _ = hidden.shape

        identity = hidden

        # projection
        q = self.query_proj(hidden)  # (b, s, d)
        k = self.key_proj(hidden)  # (b, s, d)
        v = self.value_proj(hidden)  # (b, s, d)

        # reshape
        q = self._reshape_qkv(q)  # (b, h, s, d/h)
        k = self._reshape_qkv(k)  # (b, h, s, d/h)
        v = self._reshape_qkv(v)  # (b, h, s, d/h)

        # scaled dot-product
        # (b, h, s, d/h) x (b, h, d/h, s) = (b, h, s, s)
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn *= self.attn_scale
        attn = torch.softmax(attn, dim=-1)  # (b, h, s, s)
        attn_drop = self.attn_drop(attn)

        # value multiply
        # (b, h, s, s) x (b, h, s, d/h) = (b, h, s, d/h)
        output = torch.matmul(attn_drop, v)
        output = self._reshape_output(output)  # (b, s, d)

        # output projection
        output = self.out_proj(output)
        output = self.drop(output)

        output = output + identity
        output = self.norm(output)

        return output, attn
