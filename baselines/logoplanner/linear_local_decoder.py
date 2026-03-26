import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.utils.checkpoint import checkpoint

from Pi3.pi3.models.dinov2.layers import Mlp


class LinearLocalAttention(nn.Module):
    """Kernelized linear attention with a lightweight local conv branch."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        rope=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.local_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.rope = rope

    def forward(self, x: torch.Tensor, xpos=None) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        q = self.q_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)
        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        q = F.elu(q, alpha=1.0) + 1.0
        k = F.elu(k, alpha=1.0) + 1.0

        kv_state = torch.einsum("bhnd,bhne->bhde", k, v)
        k_state = k.sum(dim=2)
        attn_out = torch.einsum("bhnd,bhde->bhne", q, kv_state)
        denom = torch.einsum("bhnd,bhd->bhn", q, k_state).unsqueeze(-1).clamp_min(1e-6)
        attn_out = (attn_out / denom).transpose(1, 2).reshape(bsz, seq_len, dim)

        local_out = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        return self.proj(attn_out + local_out)


class LinearLocalBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        rope=None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LinearLocalAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=True,
            norm_layer=norm_layer,
            rope=rope,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=ffn_bias,
        )

    def forward(self, x: torch.Tensor, xpos=None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), xpos=xpos)
        x = x + self.mlp(self.norm2(x))
        return x


class LinearLocalDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dec_embed_dim: int = 512,
        depth: int = 5,
        dec_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        rope=None,
        need_project: bool = True,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.projects = nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                LinearLocalBlock(
                    dim=dec_embed_dim,
                    num_heads=dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    act_layer=nn.GELU,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )
        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden: torch.Tensor, xpos=None) -> torch.Tensor:
        hidden = self.projects(hidden)
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, xpos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=xpos)
        return self.linear_out(hidden)
