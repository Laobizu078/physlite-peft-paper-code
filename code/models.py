"""Temporal heads and parameter-efficient adapters for timm ViT backbones."""

from __future__ import annotations

import torch
from torch import nn


class PhysDeltaHead(nn.Module):
    """Two-branch temporal head for static state and physical change."""

    def __init__(self, feature_dim: int, hidden: int = 256, num_classes: int = 2) -> None:
        super().__init__()
        self.static_branch = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
        )
        self.dynamic_branch = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.Sigmoid(),
        )
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, first: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        static = 0.5 * (first + last)
        dynamic = last - first
        static_h = self.static_branch(static)
        dynamic_h = self.dynamic_branch(dynamic)
        gate = self.gate(dynamic)
        return self.classifier(static_h + gate * dynamic_h)


class PhysResidualHead(nn.Module):
    """Standard temporal MLP plus a zero-initialized physical-delta logit branch."""

    def __init__(
        self,
        feature_dim: int,
        hidden: int = 256,
        num_classes: int = 2,
        delta_hidden: int = 128,
        delta_dropout: float = 0.0,
        residual_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.base = nn.Sequential(
            nn.LayerNorm(feature_dim * 3),
            nn.Linear(feature_dim * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )
        self.delta = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, delta_hidden),
            nn.GELU(),
            nn.Dropout(delta_dropout),
            nn.Linear(delta_hidden, num_classes),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale_init))

    def forward(self, first: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        dynamic = last - first
        base_logits = self.base(torch.cat([first, last, dynamic], dim=-1))
        return base_logits + self.delta(dynamic) * self.residual_scale


class TemporalEvidenceHead(nn.Module):
    """Tiny motion-aware temporal pooling head over frozen frame features."""

    def __init__(self, feature_dim: int, hidden: int = 256, num_classes: int = 2) -> None:
        super().__init__()
        attn_hidden = max(32, hidden // 2)
        self.frame_norm = nn.LayerNorm(feature_dim)
        self.delta_norm = nn.LayerNorm(feature_dim)
        self.attn_score = nn.Sequential(
            nn.Linear(feature_dim * 2, attn_hidden),
            nn.GELU(),
            nn.Linear(attn_hidden, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim * 5),
            nn.Linear(feature_dim * 5, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        first = feats[:, 0]
        last = feats[:, -1]
        if feats.shape[1] > 1:
            prev = torch.cat([feats[:, :1], feats[:, :-1]], dim=1)
            frame_delta = (feats - prev).abs()
            step_abs_mean = frame_delta[:, 1:].mean(dim=1)
        else:
            frame_delta = torch.zeros_like(feats)
            step_abs_mean = torch.zeros_like(first)
        score_in = torch.cat([self.frame_norm(feats), self.delta_norm(frame_delta)], dim=-1)
        weights = torch.softmax(self.attn_score(score_in).squeeze(-1), dim=1)
        pooled = torch.sum(feats * weights.unsqueeze(-1), dim=1)
        z = torch.cat([first, last, last - first, pooled, step_abs_mean], dim=-1)
        return self.classifier(z)


class FrameBackboneClassifier(nn.Module):
    """Apply an image backbone to frames, then classify a temporal aggregate."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int = 2,
        temporal: str = "motion_diff",
        head_hidden: int = 256,
        head_type: str = "mlp",
        phys_delta_hidden: int = 128,
        phys_delta_dropout: float = 0.0,
        phys_residual_scale_init: float = 1.0,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.temporal = temporal
        self.head_type = head_type
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if head_type == "phys_delta":
            self.head = PhysDeltaHead(feature_dim=feature_dim, hidden=head_hidden, num_classes=num_classes)
        elif head_type == "phys_residual":
            self.head = PhysResidualHead(
                feature_dim=feature_dim,
                hidden=head_hidden,
                num_classes=num_classes,
                delta_hidden=phys_delta_hidden,
                delta_dropout=phys_delta_dropout,
                residual_scale_init=phys_residual_scale_init,
            )
        elif temporal == "temporal_attn":
            self.head = TemporalEvidenceHead(feature_dim=feature_dim, hidden=head_hidden, num_classes=num_classes)
        elif head_type in {"mlp", "linear"}:
            if temporal == "mean":
                head_in = feature_dim
            elif temporal == "concat":
                head_in = feature_dim * 4
            elif temporal == "motion_diff":
                head_in = feature_dim * 3
            elif temporal == "motion_stats":
                head_in = feature_dim * 5
            elif temporal == "motion_bins":
                head_in = feature_dim * 7
            else:
                raise ValueError(f"Unknown temporal mode: {temporal}")

            if head_type == "linear":
                self.head = nn.Sequential(
                    nn.LayerNorm(head_in),
                    nn.Linear(head_in, num_classes),
                )
            else:
                self.head = nn.Sequential(
                    nn.LayerNorm(head_in),
                    nn.Linear(head_in, head_hidden),
                    nn.GELU(),
                    nn.Linear(head_hidden, num_classes),
                )
        else:
            raise ValueError(f"Unknown head type: {head_type}")

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: [B, T, C, H, W]
        bsz, frames, channels, height, width = video.shape
        set_lora_motion_gates(self.backbone, build_motion_gates(video))
        set_lora_temporal_context(self.backbone, build_temporal_context(video))
        flat = video.reshape(bsz * frames, channels, height, width)
        try:
            feats = self.backbone(flat)
        finally:
            clear_lora_motion_gates(self.backbone)
        feats = feats.reshape(bsz, frames, -1)

        if self.temporal == "mean":
            z = feats.mean(dim=1)
        elif self.temporal == "concat":
            z = feats[:, :4].reshape(bsz, -1)
        elif self.temporal == "temporal_attn":
            return self.head(feats)
        elif self.temporal == "motion_stats":
            first = feats[:, 0]
            last = feats[:, -1]
            mean = feats.mean(dim=1)
            if frames > 1:
                step_abs_mean = (feats[:, 1:] - feats[:, :-1]).abs().mean(dim=1)
            else:
                step_abs_mean = torch.zeros_like(first)
            z = torch.cat([first, last, last - first, mean, step_abs_mean], dim=-1)
        elif self.temporal == "motion_bins":
            first = feats[:, 0]
            last = feats[:, -1]
            mean = feats.mean(dim=1)
            if frames > 1:
                step_abs = (feats[:, 1:] - feats[:, :-1]).abs()
                split = max(1, step_abs.shape[1] // 2)
                early_abs = step_abs[:, :split].mean(dim=1)
                late_abs = step_abs[:, split:].mean(dim=1) if split < step_abs.shape[1] else early_abs
            else:
                early_abs = torch.zeros_like(first)
                late_abs = torch.zeros_like(first)
            z = torch.cat([first, last, last - first, mean, early_abs, late_abs, late_abs - early_abs], dim=-1)
        else:
            first = feats[:, 0]
            last = feats[:, -1]
            if self.head_type in {"phys_delta", "phys_residual"}:
                return self.head(first, last)
            z = torch.cat([first, last, last - first], dim=-1)
        return self.head(z)


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Return trainable and total parameter counts."""

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


class LoRAQKVLinear(nn.Module):
    """LoRA adapter for fused ViT qkv projections, updating q and v only."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
        lora_targets: str = "qv",
    ) -> None:
        super().__init__()
        if base.out_features % 3 != 0:
            raise ValueError("Expected fused qkv projection with out_features divisible by 3.")
        target_set = set(lora_targets)
        if not target_set.issubset({"q", "v"}) or not target_set:
            raise ValueError(f"Unknown LoRA targets: {lora_targets!r}")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        dim = base.out_features // 3
        self.q_a = nn.Linear(base.in_features, rank, bias=False) if "q" in target_set else None
        self.q_b = nn.Linear(rank, dim, bias=False) if "q" in target_set else None
        self.v_a = nn.Linear(base.in_features, rank, bias=False) if "v" in target_set else None
        self.v_b = nn.Linear(rank, dim, bias=False) if "v" in target_set else None
        if self.q_a is not None and self.q_b is not None:
            nn.init.kaiming_uniform_(self.q_a.weight, a=5**0.5)
            nn.init.zeros_(self.q_b.weight)
        if self.v_a is not None and self.v_b is not None:
            nn.init.kaiming_uniform_(self.v_a.weight, a=5**0.5)
            nn.init.zeros_(self.v_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        dim = out.shape[-1] // 3
        q, k, v = out.split(dim, dim=-1)
        dropped = self.dropout(x)
        if self.q_a is not None and self.q_b is not None:
            q = q + self.q_b(self.q_a(dropped)) * self.scale
        if self.v_a is not None and self.v_b is not None:
            v = v + self.v_b(self.v_a(dropped)) * self.scale
        return torch.cat([q, k, v], dim=-1)


class QueryDominantLoRAQKVLinear(nn.Module):
    """Distributed query LoRA with an optional low-rank late value correction."""

    def __init__(
        self,
        base: nn.Linear,
        query_rank: int = 4,
        query_alpha: float = 8.0,
        value_rank: int | None = None,
        value_alpha: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if base.out_features % 3 != 0:
            raise ValueError("Expected fused qkv projection with out_features divisible by 3.")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        dim = base.out_features // 3
        self.dropout = nn.Dropout(dropout)
        self.q_scale = query_alpha / query_rank
        self.q_a = nn.Linear(base.in_features, query_rank, bias=False)
        self.q_b = nn.Linear(query_rank, dim, bias=False)
        self.v_scale = None
        self.v_a = None
        self.v_b = None
        nn.init.kaiming_uniform_(self.q_a.weight, a=5**0.5)
        nn.init.zeros_(self.q_b.weight)
        if value_rank is not None:
            self.add_value_branch(value_rank, value_alpha)

    def add_value_branch(self, rank: int, alpha: float) -> None:
        """Initialize the optional value branch without changing query weights."""

        if self.v_a is not None or self.v_b is not None:
            raise RuntimeError("Value branch is already initialized.")
        dim = self.base.out_features // 3
        self.v_scale = alpha / rank
        self.v_a = nn.Linear(self.base.in_features, rank, bias=False)
        self.v_b = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.v_a.weight, a=5**0.5)
        nn.init.zeros_(self.v_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        dim = out.shape[-1] // 3
        q, k, v = out.split(dim, dim=-1)
        dropped = self.dropout(x)
        q = q + self.q_b(self.q_a(dropped)) * self.q_scale
        if self.v_a is not None and self.v_b is not None and self.v_scale is not None:
            v = v + self.v_b(self.v_a(dropped)) * self.v_scale
        return torch.cat([q, k, v], dim=-1)


class PhyGateLoRAQKVLinear(LoRAQKVLinear):
    """LoRA-qv with a per-frame motion gate for physical dynamics."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
        lora_targets: str = "qv",
    ) -> None:
        super().__init__(base=base, rank=rank, alpha=alpha, dropout=dropout, lora_targets=lora_targets)
        self.gate_scale = nn.Parameter(torch.ones(1))
        self.gate_bias = nn.Parameter(torch.zeros(1))
        self._motion_gate: torch.Tensor | None = None

    def set_motion_gate(self, gate: torch.Tensor | None) -> None:
        self._motion_gate = gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        dim = out.shape[-1] // 3
        q, k, v = out.split(dim, dim=-1)
        dropped = self.dropout(x)
        if self._motion_gate is not None:
            gate = self._motion_gate.to(device=x.device, dtype=x.dtype)
            gate = torch.sigmoid(gate * self.gate_scale.to(dtype=x.dtype) + self.gate_bias.to(dtype=x.dtype))
        else:
            gate = None
        if self.q_a is not None and self.q_b is not None:
            q_delta = self.q_b(self.q_a(dropped)) * self.scale
            q = q + (q_delta * gate if gate is not None else q_delta)
        if self.v_a is not None and self.v_b is not None:
            v_delta = self.v_b(self.v_a(dropped)) * self.scale
            v = v + (v_delta * gate if gate is not None else v_delta)
        return torch.cat([q, k, v], dim=-1)


class IA3LoRAQKVLinear(LoRAQKVLinear):
    """LoRA-qv with IA3-style output scaling for controlled adaptation."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
        targets: str = "qv",
        lora_targets: str = "qv",
    ) -> None:
        super().__init__(base=base, rank=rank, alpha=alpha, dropout=dropout, lora_targets=lora_targets)
        target_set = set(targets)
        if not target_set.issubset({"q", "k", "v"}):
            raise ValueError(f"Unknown IA3 targets: {targets!r}")
        dim = base.out_features // 3
        self.q_scale = nn.Parameter(torch.ones(dim)) if "q" in target_set else None
        self.k_scale = nn.Parameter(torch.ones(dim)) if "k" in target_set else None
        self.v_scale = nn.Parameter(torch.ones(dim)) if "v" in target_set else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        dim = out.shape[-1] // 3
        q, k, v = out.split(dim, dim=-1)
        dropped = self.dropout(x)
        if self.q_a is not None and self.q_b is not None:
            q = q + self.q_b(self.q_a(dropped)) * self.scale
        if self.v_a is not None and self.v_b is not None:
            v = v + self.v_b(self.v_a(dropped)) * self.scale
        if self.q_scale is not None:
            q = q * self.q_scale.to(dtype=out.dtype)
        if self.k_scale is not None:
            k = k * self.k_scale.to(dtype=out.dtype)
        if self.v_scale is not None:
            v = v * self.v_scale.to(dtype=out.dtype)
        return torch.cat([q, k, v], dim=-1)


class MotionValueIA3LoRAQKVLinear(LoRAQKVLinear):
    """Value-calibrated LoRA with an identity-initialized motion gate.

    The gate only modulates value-branch LoRA updates. Query routing remains the
    same as ordinary LoRA, and the gate starts as an exact no-op.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
        targets: str = "v",
        lora_targets: str = "qv",
    ) -> None:
        super().__init__(base=base, rank=rank, alpha=alpha, dropout=dropout, lora_targets=lora_targets)
        target_set = set(targets)
        if not target_set.issubset({"q", "k", "v"}):
            raise ValueError(f"Unknown IA3 targets: {targets!r}")
        dim = base.out_features // 3
        self.q_scale = nn.Parameter(torch.ones(dim)) if "q" in target_set else None
        self.k_scale = nn.Parameter(torch.ones(dim)) if "k" in target_set else None
        self.v_scale = nn.Parameter(torch.ones(dim)) if "v" in target_set else None
        self.motion_delta_scale = nn.Parameter(torch.zeros(1))
        self.motion_delta_bias = nn.Parameter(torch.zeros(1))
        self._motion_gate: torch.Tensor | None = None

    def set_motion_gate(self, gate: torch.Tensor | None) -> None:
        self._motion_gate = gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        dim = out.shape[-1] // 3
        q, k, v = out.split(dim, dim=-1)
        dropped = self.dropout(x)
        if self.q_a is not None and self.q_b is not None:
            q = q + self.q_b(self.q_a(dropped)) * self.scale
        if self.v_a is not None and self.v_b is not None:
            v_delta = self.v_b(self.v_a(dropped)) * self.scale
            if self._motion_gate is not None:
                gate = self._motion_gate.to(device=x.device, dtype=x.dtype)
                gate = torch.tanh(gate + self.motion_delta_bias.to(dtype=x.dtype))
                v_delta = v_delta * (1.0 + self.motion_delta_scale.to(dtype=x.dtype) * gate)
            v = v + v_delta
        if self.q_scale is not None:
            q = q * self.q_scale.to(dtype=out.dtype)
        if self.k_scale is not None:
            k = k * self.k_scale.to(dtype=out.dtype)
        if self.v_scale is not None:
            v = v * self.v_scale.to(dtype=out.dtype)
        return torch.cat([q, k, v], dim=-1)


class TemporalConditionedLoRAQKVLinear(LoRAQKVLinear):
    """LoRA whose value update is conditioned on temporal phase and motion.

    The channel-wise gate is zero-initialized, so this module starts exactly as
    ordinary LoRA. Query routing remains unconditioned; only injected value
    evidence changes across frames.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
        lora_targets: str = "qv",
    ) -> None:
        super().__init__(base=base, rank=rank, alpha=alpha, dropout=dropout, lora_targets=lora_targets)
        dim = base.out_features // 3
        self.context_proj = nn.Linear(3, dim, bias=False)
        nn.init.zeros_(self.context_proj.weight)
        self._temporal_context: torch.Tensor | None = None

    def set_temporal_context(self, context: torch.Tensor | None) -> None:
        self._temporal_context = context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        dim = out.shape[-1] // 3
        q, k, v = out.split(dim, dim=-1)
        dropped = self.dropout(x)
        if self.q_a is not None and self.q_b is not None:
            q = q + self.q_b(self.q_a(dropped)) * self.scale
        if self.v_a is not None and self.v_b is not None:
            v_delta = self.v_b(self.v_a(dropped)) * self.scale
            if self._temporal_context is not None:
                context = self._temporal_context.to(device=x.device, dtype=x.dtype)
                gate = 1.0 + torch.tanh(self.context_proj(context))
                v_delta = v_delta * gate
            v = v + v_delta
        return torch.cat([q, k, v], dim=-1)


class SSFLayerNorm(nn.Module):
    """Scaling and shifting features on top of a frozen LayerNorm."""

    def __init__(self, base: nn.LayerNorm) -> None:
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        shape = base.normalized_shape
        self.ssf_scale = nn.Parameter(torch.ones(shape))
        self.ssf_shift = nn.Parameter(torch.zeros(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) * self.ssf_scale + self.ssf_shift


class ResidualAdapter(nn.Module):
    """Small residual bottleneck adapter after a frozen transformer block."""

    def __init__(self, dim: int, bottleneck: int = 64, init_scale: float = 1e-3) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.scale = nn.Parameter(torch.tensor(init_scale))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(self.norm(x)))) * self.scale


class AdaptFormerMLP(nn.Module):
    """AdaptFormer-style parallel bottleneck branch for a frozen ViT MLP."""

    def __init__(
        self,
        original_mlp: nn.Module,
        in_dim: int,
        bottleneck: int = 64,
        scale: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.original_mlp = original_mlp
        for param in self.original_mlp.parameters():
            param.requires_grad = False
        self.down_proj = nn.Linear(in_dim, bottleneck)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck, in_dim)
        self.scale = scale
        nn.init.kaiming_uniform_(self.down_proj.weight, a=5**0.5)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapted = self.up_proj(self.dropout(self.act(self.down_proj(x))))
        return self.original_mlp(x) + adapted * self.scale


class ShallowVisualPrompt(nn.Module):
    """VPT-shallow wrapper for timm ViT backbones."""

    def __init__(self, base: nn.Module, prompt_tokens: int = 8, init_std: float = 0.02) -> None:
        super().__init__()
        if not hasattr(base, "patch_embed") or not hasattr(base, "blocks"):
            raise RuntimeError("Visual prompt tuning requires a ViT-style timm backbone.")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.num_features = getattr(base, "num_features", None)
        if self.num_features is None:
            raise RuntimeError("Could not infer VPT feature dimension from backbone.num_features.")
        self.prompt = nn.Parameter(torch.empty(1, prompt_tokens, self.num_features))
        nn.init.normal_(self.prompt, std=init_std)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base.patch_embed(x)
        x = self.base._pos_embed(x)
        prefix_tokens = int(getattr(self.base, "num_prefix_tokens", 1))
        prompt = self.prompt.expand(x.shape[0], -1, -1).to(dtype=x.dtype)
        x = torch.cat([x[:, :prefix_tokens], prompt, x[:, prefix_tokens:]], dim=1)
        x = self.base.patch_drop(x)
        x = self.base.norm_pre(x)
        x = self.base.blocks(x)
        x = self.base.norm(x)
        return torch.cat([x[:, :prefix_tokens], x[:, prefix_tokens + self.prompt.shape[1] :]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.forward_head(self.forward_features(x))


class IA3QKVLinear(nn.Module):
    """IA3-style activation scaling for frozen fused qkv projections."""

    def __init__(self, base: nn.Linear, tune_query: bool = True, tune_key: bool = False, tune_value: bool = True) -> None:
        super().__init__()
        if base.out_features % 3 != 0:
            raise ValueError("Expected fused qkv projection with out_features divisible by 3.")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        dim = base.out_features // 3
        self.q_scale = nn.Parameter(torch.ones(dim)) if tune_query else None
        self.k_scale = nn.Parameter(torch.ones(dim)) if tune_key else None
        self.v_scale = nn.Parameter(torch.ones(dim)) if tune_value else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        dim = out.shape[-1] // 3
        q, k, v = out.split(dim, dim=-1)
        if self.q_scale is not None:
            q = q * self.q_scale.to(dtype=out.dtype)
        if self.k_scale is not None:
            k = k * self.k_scale.to(dtype=out.dtype)
        if self.v_scale is not None:
            v = v * self.v_scale.to(dtype=out.dtype)
        return torch.cat([q, k, v], dim=-1)


class AdapterBlock(nn.Module):
    """Wrap a frozen ViT block with a trainable residual adapter."""

    def __init__(self, base: nn.Module, dim: int, bottleneck: int = 64) -> None:
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.adapter = ResidualAdapter(dim=dim, bottleneck=bottleneck)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.base(x))


def build_motion_gates(video: torch.Tensor) -> torch.Tensor:
    """Return [B*T, 1, 1] standardized frame-motion gates."""

    bsz, frames = video.shape[:2]
    if frames <= 1:
        motion = torch.zeros((bsz, frames), device=video.device, dtype=video.dtype)
    else:
        diffs = (video[:, 1:] - video[:, :-1]).abs().mean(dim=(2, 3, 4))
        motion = torch.cat([diffs[:, :1], diffs], dim=1)
    motion = (motion - motion.mean(dim=1, keepdim=True)) / (motion.std(dim=1, keepdim=True, unbiased=False) + 1e-6)
    return motion.reshape(bsz * frames, 1, 1)


def build_temporal_context(video: torch.Tensor) -> torch.Tensor:
    """Return [B*T, 1, 3] phase, motion, and phase-motion context."""

    bsz, frames = video.shape[:2]
    phase = torch.linspace(-1.0, 1.0, frames, device=video.device, dtype=video.dtype)
    phase = phase.unsqueeze(0).expand(bsz, -1)
    if frames <= 1:
        motion = torch.zeros_like(phase)
    else:
        diffs = (video[:, 1:] - video[:, :-1]).abs().mean(dim=(2, 3, 4))
        motion = torch.cat([diffs[:, :1], diffs], dim=1)
        motion = (motion - motion.mean(dim=1, keepdim=True)) / (
            motion.std(dim=1, keepdim=True, unbiased=False) + 1e-6
        )
    context = torch.stack([phase, motion, phase * motion], dim=-1)
    return context.reshape(bsz * frames, 1, 3)


def set_lora_motion_gates(backbone: nn.Module, gates: torch.Tensor) -> None:
    """Attach per-frame motion gates to motion-aware LoRA wrappers."""

    for module in backbone.modules():
        if isinstance(module, (PhyGateLoRAQKVLinear, MotionValueIA3LoRAQKVLinear)):
            module.set_motion_gate(gates)


def set_lora_temporal_context(backbone: nn.Module, context: torch.Tensor) -> None:
    """Attach temporal context to conditioned LoRA wrappers."""

    for module in backbone.modules():
        if isinstance(module, TemporalConditionedLoRAQKVLinear):
            module.set_temporal_context(context)


def clear_lora_motion_gates(backbone: nn.Module) -> None:
    """Clear transient motion and temporal context after a forward pass."""

    for module in backbone.modules():
        if isinstance(module, (PhyGateLoRAQKVLinear, MotionValueIA3LoRAQKVLinear)):
            module.set_motion_gate(None)
        if isinstance(module, TemporalConditionedLoRAQKVLinear):
            module.set_temporal_context(None)


def _select_tail_modules(items: list, layers: str) -> list:
    """Select all, trailing, or explicitly indexed modules."""

    if layers == "all":
        return items
    if layers == "last_half":
        return items[len(items) // 2 :]
    if layers.startswith("last"):
        count = int(layers.removeprefix("last"))
        return items[-count:]
    if layers.startswith("idx:"):
        raw_indices = layers.removeprefix("idx:")
        if not raw_indices:
            raise ValueError("Layer selector 'idx:' must include at least one index.")
        indices = [int(idx.strip()) for idx in raw_indices.split(",") if idx.strip()]
        try:
            return [items[idx] for idx in indices]
        except IndexError as exc:
            raise ValueError(f"Layer selector {layers!r} is out of range for {len(items)} modules.") from exc
    raise ValueError(f"Unknown LoRA layer selector: {layers}")


def apply_lora_qv_to_vit(
    backbone: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    layers: str = "last4",
    phygate: bool = False,
    ia3_targets: str | None = None,
    motion_value_gate: bool = False,
    temporal_conditioning: bool = False,
    lora_targets: str = "qv",
) -> int:
    """Replace timm ViT fused qkv projections with q/v LoRA adapters."""

    qkv_modules: list[tuple[nn.Module, str, nn.Linear]] = []
    for parent in backbone.modules():
        for child_name, child in parent.named_children():
            if child_name == "qkv" and isinstance(child, nn.Linear):
                qkv_modules.append((parent, child_name, child))
    selected = _select_tail_modules(qkv_modules, layers)
    if not selected:
        raise RuntimeError("No fused qkv Linear modules found for LoRA injection.")
    if sum(bool(flag) for flag in [phygate, ia3_targets is not None, motion_value_gate, temporal_conditioning]) > 1:
        raise ValueError("PhyGate, IA3-LoRA, Motion-Value LoRA, and temporal LoRA are separate wrappers.")
    if temporal_conditioning:
        lora_cls = TemporalConditionedLoRAQKVLinear
    elif motion_value_gate:
        lora_cls = MotionValueIA3LoRAQKVLinear
    elif ia3_targets is not None:
        lora_cls = IA3LoRAQKVLinear
    elif phygate:
        lora_cls = PhyGateLoRAQKVLinear
    else:
        lora_cls = LoRAQKVLinear
    for parent, child_name, child in selected:
        kwargs = {
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "lora_targets": lora_targets,
        }
        if ia3_targets is not None or motion_value_gate:
            kwargs["targets"] = ia3_targets or "v"
        setattr(
            parent,
            child_name,
            lora_cls(
                child,
                **kwargs,
            ),
        )
    return len(selected)


def apply_query_dominant_lora_to_vit(
    backbone: nn.Module,
    query_rank: int = 4,
    query_alpha: float = 8.0,
    query_layers: str = "last8",
    value_rank: int = 1,
    value_alpha: float = 2.0,
    value_layers: str = "last4",
    dropout: float = 0.0,
) -> int:
    """Inject query LoRA broadly and a smaller value correction near output."""

    qkv_modules: list[tuple[nn.Module, str, nn.Linear]] = []
    for parent in backbone.modules():
        for child_name, child in parent.named_children():
            if child_name == "qkv" and isinstance(child, nn.Linear):
                qkv_modules.append((parent, child_name, child))
    query_selected = _select_tail_modules(qkv_modules, query_layers)
    value_ids = {id(child) for _, _, child in _select_tail_modules(qkv_modules, value_layers)}
    if not query_selected:
        raise RuntimeError("No fused qkv Linear modules found for query-dominant LoRA injection.")
    if not value_ids.issubset({id(child) for _, _, child in query_selected}):
        raise ValueError("Value-correction layers must be a subset of query-adapted layers.")
    wrappers = {}
    for parent, child_name, child in query_selected:
        wrapper = QueryDominantLoRAQKVLinear(
            child,
            query_rank=query_rank,
            query_alpha=query_alpha,
            value_rank=None,
            dropout=dropout,
        )
        setattr(
            parent,
            child_name,
            wrapper,
        )
        wrappers[id(child)] = wrapper
    for child_id in value_ids:
        wrappers[child_id].add_value_branch(value_rank, value_alpha)
    return len(query_selected)


def apply_ssf_to_layernorm(backbone: nn.Module, layers: str = "all") -> int:
    """Wrap selected LayerNorm modules with trainable scale and shift."""

    layer_norms: list[tuple[nn.Module, str, nn.LayerNorm]] = []
    for parent in backbone.modules():
        for child_name, child in parent.named_children():
            if isinstance(child, nn.LayerNorm):
                layer_norms.append((parent, child_name, child))
    selected = _select_tail_modules(layer_norms, layers)
    if not selected:
        raise RuntimeError("No LayerNorm modules found for SSF injection.")
    for parent, child_name, child in selected:
        setattr(parent, child_name, SSFLayerNorm(child))
    return len(selected)


def apply_adapters_to_vit(backbone: nn.Module, bottleneck: int = 64, layers: str = "last4") -> int:
    """Append residual bottleneck adapters to selected ViT blocks."""

    if not hasattr(backbone, "blocks"):
        raise RuntimeError("Backbone has no ViT-style blocks for adapter injection.")
    blocks = list(backbone.blocks)
    selected = _select_tail_modules([(idx, block) for idx, block in enumerate(blocks)], layers)
    if not selected:
        raise RuntimeError("No ViT blocks selected for adapter injection.")
    dim = getattr(backbone, "num_features", None)
    if dim is None:
        raise RuntimeError("Could not infer adapter feature dimension from backbone.num_features.")
    for idx, block in selected:
        backbone.blocks[idx] = AdapterBlock(block, dim=dim, bottleneck=bottleneck)
    return len(selected)


def apply_adaptformer_to_vit(
    backbone: nn.Module,
    bottleneck: int = 64,
    layers: str = "last4",
    scale: float = 0.1,
    dropout: float = 0.0,
) -> int:
    """Replace selected ViT MLPs with AdaptFormer parallel MLP branches."""

    if not hasattr(backbone, "blocks"):
        raise RuntimeError("Backbone has no ViT-style blocks for AdaptFormer injection.")
    blocks = list(backbone.blocks)
    selected = _select_tail_modules([(idx, block) for idx, block in enumerate(blocks)], layers)
    if not selected:
        raise RuntimeError("No ViT blocks selected for AdaptFormer injection.")
    dim = getattr(backbone, "num_features", None)
    if dim is None:
        raise RuntimeError("Could not infer AdaptFormer feature dimension from backbone.num_features.")
    for idx, block in selected:
        if not hasattr(block, "mlp"):
            raise RuntimeError(f"Block {idx} has no MLP module for AdaptFormer injection.")
        block.mlp = AdaptFormerMLP(
            block.mlp,
            in_dim=dim,
            bottleneck=bottleneck,
            scale=scale,
            dropout=dropout,
        )
    return len(selected)


def apply_vpt_to_vit(backbone: nn.Module, prompt_tokens: int = 8, init_std: float = 0.02) -> ShallowVisualPrompt:
    """Wrap a ViT backbone with shallow visual prompt tokens."""

    return ShallowVisualPrompt(backbone, prompt_tokens=prompt_tokens, init_std=init_std)


def apply_ia3_to_vit(backbone: nn.Module, layers: str = "last4", targets: str = "qv") -> int:
    """Replace selected fused qkv projections with IA3-style output scalers."""

    qkv_modules: list[tuple[nn.Module, str, nn.Linear]] = []
    for parent in backbone.modules():
        for child_name, child in parent.named_children():
            if child_name == "qkv" and isinstance(child, nn.Linear):
                qkv_modules.append((parent, child_name, child))
    selected = _select_tail_modules(qkv_modules, layers)
    if not selected:
        raise RuntimeError("No fused qkv Linear modules found for IA3 injection.")
    target_set = set(targets)
    if not target_set.issubset({"q", "k", "v"}):
        raise ValueError(f"Unknown IA3 targets: {targets!r}")
    for parent, child_name, child in selected:
        setattr(
            parent,
            child_name,
            IA3QKVLinear(
                child,
                tune_query="q" in target_set,
                tune_key="k" in target_set,
                tune_value="v" in target_set,
            ),
        )
    return len(selected)


def enable_bitfit(backbone: nn.Module) -> int:
    """Enable only backbone bias parameters and return their tensor count."""

    count = 0
    for name, param in backbone.named_parameters():
        if name.endswith(".bias"):
            param.requires_grad = True
            count += 1
    return count
