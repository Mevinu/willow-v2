"""Qwen3-MoE and Qwen3.5/3.6-MoE: how their layers are wired, on the shared pieces in engines/base.py."""
import time

import torch
import torch.nn.functional as F

from engines.base import Engine, GatedDeltaNet, QwenChat

class MoEEngine(Engine):
    """Qwen3-MoE: each layer is RMSNorm -> attention -> residual add, then RMSNorm -> MoE -> residual add."""

    @torch.no_grad()
    def forward(self, token_ids: list) -> torch.Tensor:
        self.ops.decoding = len(token_ids) == 1
        x = self.embed(token_ids)
        cos, sin = self.rope(len(token_ids))
        for layer in range(self.num_layers):
            p = f"{self.prefix}layers.{layer}"
            t = time.perf_counter()
            x = x + self.token_mixer(self.rms_norm(x, p + ".input_layernorm.weight"), layer, cos, sin)
            self.tick(self.layer_types[layer].replace("_", " "), t)
            x = x + self.moe(self.rms_norm(x, p + ".post_attention_layernorm.weight"), layer, residual=x)
        self.position += len(token_ids)
        t = time.perf_counter()
        logits = F.linear(self.rms_norm(x, self.prefix + "norm.weight"), self.w("lm_head.weight"))
        self.tick("lm_head", t)
        return logits

    def router_guess_input(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        return self.rms_norm(residual, f"{self.prefix}layers.{layer}.post_attention_layernorm.weight")

class Qwen35MoEEngine(GatedDeltaNet, QwenChat, MoEEngine):
    """Qwen3.5/3.6-MoE: hybrid Gated DeltaNet (linear attention) + gated full attention, plus a shared expert."""

    prefix = "model.language_model."
    zero_centered_norm = True

    def setup(self, cfg: dict):
        super().setup(cfg)
        self.norm_topk_prob = True  # Qwen3.5's router always renormalizes the top-k weights
        self.setup_linear_attention(cfg)

    def reset(self):
        super().reset()
        self.reset_linear_attention()

    def token_mixer(self, x: torch.Tensor, layer: int, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if self.layer_types[layer] == "linear_attention":
            return self.linear_attention(x, layer)
        return self.attention(x, layer, cos, sin)
