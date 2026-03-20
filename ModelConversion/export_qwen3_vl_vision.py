#!/usr/bin/env python3
"""Export the real Qwen3-VL-2B vision tower to TorchScript.

This script is designed to resume from the local Hugging Face cache if the
checkpoint download is interrupted. It exports only the vision side to a fixed
1536-dim embedding so it can slot into the current iOS retrieval pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)


class VisionTowerWrapper(torch.nn.Module):
    def __init__(self, visual: torch.nn.Module, output_dim: int) -> None:
        super().__init__()
        self.visual = visual
        self.output_dim = output_dim

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        image_tokens, _ = self.visual(pixel_values, grid_thw=grid_thw)
        pooled = image_tokens.mean(dim=0, keepdim=True)
        pooled = pooled[..., : self.output_dim]
        return F.normalize(pooled, dim=-1)


def patch_visual_attention_for_single_sequence(visual: torch.nn.Module) -> None:
    def patched_forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            is_causal=False,
            **kwargs,
        )[0]

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output

    for block in visual.blocks:
        block.attn.forward = patched_forward.__get__(block.attn, type(block.attn))


def patch_visual_forward_for_fixed_grid(visual: torch.nn.Module, grid_thw: torch.Tensor) -> None:
    with torch.no_grad():
        pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)
        rotary_pos_emb = visual.rot_pos_emb(grid_thw)
        seq_len, _ = pos_embeds.shape
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    def patched_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        hidden_states = self.patch_embed(hidden_states)
        local_pos = pos_embeds.to(device=hidden_states.device, dtype=hidden_states.dtype)
        local_cos = cos.to(device=hidden_states.device, dtype=hidden_states.dtype)
        local_sin = sin.to(device=hidden_states.device, dtype=hidden_states.dtype)
        local_cu = cu_seqlens.to(device=hidden_states.device)

        hidden_states = hidden_states + local_pos
        position_embeddings = (local_cos, local_sin)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=local_cu,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.merger(hidden_states)
        return hidden_states, []

    visual.forward = patched_forward.__get__(visual, type(visual))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--input_resolution", type=int, default=448)
    parser.add_argument("--output_dim", type=int, default=1536)
    parser.add_argument("--cache_dir", type=Path, default=Path.home() / ".cache" / "huggingface" / "hub")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=str(args.cache_dir),
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=str(args.cache_dir))
    patch_visual_attention_for_single_sequence(model.visual)

    wrapper = VisionTowerWrapper(model.visual, args.output_dim).eval()
    image = Image.fromarray((np.random.rand(args.input_resolution, args.input_resolution, 3) * 255).astype("uint8"))
    processed = processor(text=["describe"], images=[image], return_tensors="pt")
    pixel_values = processed["pixel_values"].to(dtype=torch.float32)
    grid_thw = processed["image_grid_thw"]
    patch_visual_forward_for_fixed_grid(model.visual, grid_thw)

    traced = torch.jit.trace(wrapper, (pixel_values, grid_thw))
    output_path = args.output_dir / "qwen3_vl_vision_encoder.pt"
    traced.save(str(output_path))
    print(f"Saved TorchScript vision tower to {output_path}")


if __name__ == "__main__":
    main()
