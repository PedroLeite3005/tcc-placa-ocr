"""SVTR-Tiny para reconhecimento de placas veiculares com cabeça CTC.

Arquitetura conforme o paper:
  "SVTR: Scene Text Recognition with a Single Visual Model" (Du et al., 2022)

Dimensões para entrada 256×64:
  Patch embed (stride 4):  → (B, 16×64, 64)
  Stage 1 (3 blocos):      → merge → (B, 8×64, 128)
  Stage 2 (6 blocos):      → merge → (B, 4×64, 256)
  Stage 3 (3 blocos):      → avg H → (B, 64, 256)
  Cabeça CTC:              → (B, T=64, num_classes)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Inicialização
# ---------------------------------------------------------------------------

def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    def norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    a, b = -2 * std, 2 * std
    with torch.no_grad():
        l = norm_cdf((a - 0) / std)
        u = norm_cdf((b - 0) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.clamp_(min=a, max=b)
    return tensor


# ---------------------------------------------------------------------------
# Componentes base
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device).bernoulli_(keep).div_(keep)
        return x * mask


class LocalMixer(nn.Module):
    """Mistura local via convolução depthwise no espaço 2-D."""

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size,
                            padding=kernel_size // 2, groups=dim)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x2 = x.reshape(B, H, W, C).permute(0, 3, 1, 2)   # (B,C,H,W)
        x2 = self.bn(self.dw(x2))
        return x2.permute(0, 2, 3, 1).reshape(B, N, C)    # (B,N,C)


class GlobalMixer(nn.Module):
    """Mistura global via MHSA."""

    def __init__(self, dim: int, num_heads: int,
                 attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, H: int = 0, W: int = 0) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class MixingBlock(nn.Module):
    """Bloco de mistura SVTR: local ou global + FFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mixer: str = "global",
        H: int = 0,
        W: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.H, self.W = H, W
        self.mixer_type = mixer

        if mixer == "local":
            self.mixer = LocalMixer(dim)
        else:
            self.mixer = GlobalMixer(dim, num_heads, drop, drop)

        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_dim, dim), nn.Dropout(drop),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.mixer(self.norm1(x), self.H, self.W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Patch embedding e merge
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Embedding sobreposto em duas convoluções stride-2.

    (B, 3, H, W) → (B, H/4 * W/4, embed_dim)
    """

    def __init__(self, in_ch: int = 3, embed_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)                      # (B, C, H/4, W/4)
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2), H, W  # (B, H/4*W/4, C)


class MergeBlock(nn.Module):
    """Reduz H pela metade (stride (2,1)), mantém W.

    (B, H*W, in_dim) → (B, H/2*W, out_dim)
    """

    def __init__(self, in_dim: int, out_dim: int, H: int, W: int) -> None:
        super().__init__()
        self.H, self.W = H, W
        self.conv = nn.Conv2d(
            in_dim, out_dim,
            kernel_size=(3, 1), stride=(2, 1), padding=(1, 0),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        B, _, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W


# ---------------------------------------------------------------------------
# Modelo principal
# ---------------------------------------------------------------------------

class SVTRTiny(nn.Module):
    """SVTR-Tiny para reconhecimento de placas (CTC).

    Args:
        img_h: altura da imagem de entrada (padrão 64).
        img_w: largura da imagem de entrada (padrão 256).
        in_ch: canais de entrada (padrão 3).
        embed_dims: dimensões de cada estágio.
        depths: número de blocos por estágio.
        num_heads: cabeças de atenção por estágio.
        num_classes: tamanho do vocabulário CTC incluindo blank.
    """

    def __init__(
        self,
        img_h: int = 64,
        img_w: int = 256,
        in_ch: int = 3,
        embed_dims: tuple[int, ...] = (64, 128, 256),
        depths: tuple[int, ...] = (3, 6, 3),
        num_heads: tuple[int, ...] = (2, 4, 8),
        num_classes: int = 37,
    ) -> None:
        super().__init__()

        H0, W0 = img_h // 4, img_w // 4    # 16, 64  após patch embed

        self.patch_embed = PatchEmbed(in_ch, embed_dims[0])

        # Estágio 1: metade local + metade global
        n_local1 = depths[0] // 2
        types1 = ["local"] * n_local1 + ["global"] * (depths[0] - n_local1)
        self.stage1 = nn.ModuleList([
            MixingBlock(embed_dims[0], num_heads[0], types1[i], H0, W0)
            for i in range(depths[0])
        ])
        self.merge1 = MergeBlock(embed_dims[0], embed_dims[1], H0, W0)
        H1, W1 = H0 // 2, W0               # 8, 64

        # Estágio 2: metade local + metade global
        n_local2 = depths[1] // 2
        types2 = ["local"] * n_local2 + ["global"] * (depths[1] - n_local2)
        self.stage2 = nn.ModuleList([
            MixingBlock(embed_dims[1], num_heads[1], types2[i], H1, W1)
            for i in range(depths[1])
        ])
        self.merge2 = MergeBlock(embed_dims[1], embed_dims[2], H1, W1)
        H2, W2 = H1 // 2, W1               # 4, 64

        # Estágio 3: apenas global
        self.stage3 = nn.ModuleList([
            MixingBlock(embed_dims[2], num_heads[2], "global", H2, W2)
            for _ in range(depths[2])
        ])
        self.H2, self.W2 = H2, W2

        self.norm = nn.LayerNorm(embed_dims[2])
        self.head = nn.Linear(embed_dims[2], num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _trunc_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            logits: (B, T, num_classes)  onde T = W // 4
        """
        x, H, W = self.patch_embed(x)

        for blk in self.stage1:
            x = blk(x)
        x, H, W = self.merge1(x)

        for blk in self.stage2:
            x = blk(x)
        x, H, W = self.merge2(x)

        for blk in self.stage3:
            x = blk(x)

        # Média sobre a dimensão H → (B, W, C)
        B, _, C = x.shape
        x = x.reshape(B, self.H2, self.W2, C).mean(dim=1)  # (B, 64, 256)
        x = self.norm(x)
        return self.head(x)                                  # (B, 64, num_classes)
