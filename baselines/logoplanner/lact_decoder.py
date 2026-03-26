import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from Pi3.pi3.models.dinov2.layers import Mlp


def silu_backprop(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    sigma = torch.sigmoid(x)
    return dy * sigma * (1 + x * (1 - sigma))


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    x_type = x.dtype
    ret = x / (x.norm(dim=-1, keepdim=True) + 1e-5)
    return ret.type(x_type)


def zeropower_via_newtonschulz5(g: torch.Tensor) -> torch.Tensor:
    """
    Official LaCT repo's Newton-Schulz Muon projection, adapted without
    torch.compile decoration for easier integration here.
    """
    assert len(g.shape) == 3
    x = g.bfloat16()
    if g.size(1) > g.size(2):
        x = x.transpose(1, 2)
    x = x / (x.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        a_mat = x @ x.transpose(1, 2)
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x
    if g.size(1) > g.size(2):
        x = x.transpose(1, 2)
    return x


def bidirectional_lact_swiglu(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    use_muon: bool = True,
) -> torch.Tensor:
    """
    Adapted from the official LaCT minimal implementation:
    https://github.com/a1600012888/LaCT/blob/main/minimal_implementations/bidirectional_lact_layer.py
    """
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    q_t = q.transpose(1, 2)
    v_t = v.transpose(1, 2)

    gate_before_act = torch.bmm(w0, k.transpose(1, 2))
    hidden_before_mul = torch.bmm(w2, k.transpose(1, 2))
    hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

    dhidden = torch.bmm(w1.transpose(1, 2), v_t)
    dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
    dgate = dhidden * hidden_before_mul
    dgate_before_act = silu_backprop(dgate, gate_before_act)

    dw1 = torch.bmm(v_t, (hidden.transpose(1, 2) * lr1).type_as(v_t))
    dw0 = torch.bmm(dgate_before_act, (k * lr0).type_as(dgate_before_act))
    dw2 = torch.bmm(dhidden_before_mul, (k * lr2).type_as(dhidden_before_mul))

    if use_muon:
        dw0 = zeropower_via_newtonschulz5(dw0)
        dw1 = zeropower_via_newtonschulz5(dw1)
        dw2 = zeropower_via_newtonschulz5(dw2)

    w0 = w0 + dw0
    w1 = w1 + dw1
    w2 = w2 + dw2

    w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
    w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
    w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

    h = torch.bmm(w2, q_t)
    gate = F.silu(torch.bmm(w0, q_t), inplace=False)
    o = torch.bmm(w1, gate * h).transpose(1, 2)
    return o


def inv_softplus(x):
    if isinstance(x, torch.Tensor):
        return x + torch.log(-torch.expm1(-x))
    return x + math.log(-math.expm1(-x))


class OfficialBidirectionalLaCTSwiGLU(nn.Module):
    """
    Official bidirectional LaCT minimal implementation adapted to accept an
    optional RoPE hook so it can serve as a drop-in decoder block here.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        inter_multi: float = 1.0,
        use_o_norm: bool = True,
        qk_l2_norm: bool = True,
        use_muon: bool = True,
        base_lr: float = 1e-2,
        rope=None,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.inter_multi = inter_multi
        self.use_o_norm = use_o_norm
        self.qk_l2_norm = qk_l2_norm
        self.use_muon = use_muon
        self.rope = rope

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.lr_dim = 1
        self.lr_proj = nn.Linear(dim, self.lr_dim * 3 * self.num_heads, bias=False)
        self.base_lr = base_lr
        self.base_lr_inv = inv_softplus(base_lr)

        d_in, d_out = self.head_dim, self.head_dim
        d_h = int(self.head_dim * self.inter_multi)
        self.w0 = nn.Parameter(torch.randn(self.num_heads, d_h, d_in) / math.sqrt(d_in))
        self.w1 = nn.Parameter(torch.randn(self.num_heads, d_out, d_h) / math.sqrt(d_h))
        self.w2 = nn.Parameter(torch.randn(self.num_heads, d_h, d_in) / math.sqrt(d_in))

        self.o_norm = nn.RMSNorm(head_dim, eps=1e-5, elementwise_affine=True) if use_o_norm else nn.Identity()

    def forward(self, x: torch.Tensor, xpos=None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = F.silu(self.to_qkv(x), inplace=False)
        qkv = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_l2_norm:
            q = l2_norm(q)
            k = l2_norm(k)

        if self.rope is not None and xpos is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        q = q.reshape(bsz * self.num_heads, seq_len, self.head_dim)
        k = k.reshape(bsz * self.num_heads, seq_len, self.head_dim)
        v = v.reshape(bsz * self.num_heads, seq_len, self.head_dim)

        lr = self.lr_proj(x.float())
        lr = F.softplus(lr + self.base_lr_inv)
        lr = lr.view(bsz, seq_len, self.num_heads, 3, self.lr_dim).permute(3, 0, 2, 1, 4)
        lr0 = lr[0].reshape(bsz * self.num_heads, seq_len, self.lr_dim)
        lr1 = lr[1].reshape(bsz * self.num_heads, seq_len, self.lr_dim)
        lr2 = lr[2].reshape(bsz * self.num_heads, seq_len, self.lr_dim)

        w0 = self.w0.repeat(bsz, 1, 1)
        w1 = self.w1.repeat(bsz, 1, 1)
        w2 = self.w2.repeat(bsz, 1, 1)

        output = bidirectional_lact_swiglu(w0, w1, w2, q, k, v, lr0, lr1, lr2, self.use_muon)
        output = self.o_norm(output)
        output = output.view(bsz, self.num_heads, seq_len, self.head_dim).permute(0, 2, 1, 3).reshape(bsz, seq_len, self.dim)
        return self.o_proj(output)


class LaCTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        rope=None,
        base_lr: float = 1e-2,
        use_muon: bool = True,
    ):
        super().__init__()
        head_dim = dim // num_heads
        self.norm1 = norm_layer(dim)
        self.lact = OfficialBidirectionalLaCTSwiGLU(
            dim=dim,
            head_dim=head_dim,
            inter_multi=mlp_ratio / 4.0,
            use_o_norm=True,
            qk_l2_norm=True,
            use_muon=use_muon,
            base_lr=base_lr,
            rope=rope,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=True,
        )

    def forward(self, x: torch.Tensor, xpos=None) -> torch.Tensor:
        x = x + self.lact(self.norm1(x), xpos=xpos)
        x = x + self.mlp(self.norm2(x))
        return x


class LaCTDecoder(nn.Module):
    """
    Decoder wrapper that keeps the old GeometryModel decoder interface while
    using the official LaCT bidirectional layer as its core block.
    """

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
        base_lr: float = 1e-2,
        use_muon: bool = True,
        **_: dict,
    ) -> None:
        super().__init__()
        self.projects = nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                LaCTBlock(
                    dim=dec_embed_dim,
                    num_heads=dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    act_layer=nn.GELU,
                    rope=rope,
                    base_lr=base_lr,
                    use_muon=use_muon,
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
