#!/usr/bin/env python3
"""Export a real Qwen3-VL text-side fusion encoder to TorchScript.

This is not the full autoregressive generator. Instead it uses the real Qwen3
language-side embedding table and RMSNorm weights to produce a pooled 1536-dim
text representation that can be converted reliably to Core ML and used as the
next step beyond the placeholder text-fusion package.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration


class TextFusionWrapper(torch.nn.Module):
    def __init__(self, embed_tokens: torch.nn.Module, norm: torch.nn.Module, output_dim: int, max_tokens: int) -> None:
        super().__init__()
        self.embed_tokens = embed_tokens
        self.norm = norm
        self.output_dim = output_dim
        self.max_tokens = max_tokens

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.embed_tokens(input_ids)
        mask = attention_mask.unsqueeze(-1).to(dtype=embeddings.dtype)
        masked = embeddings * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = masked.sum(dim=1) / denom
        pooled = self.norm(pooled)
        pooled = pooled[..., : self.output_dim]
        return F.normalize(pooled, dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_dim", type=int, default=1536)
    parser.add_argument("--max_tokens", type=int, default=32)
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

    wrapper = TextFusionWrapper(
        model.model.language_model.embed_tokens,
        model.model.language_model.norm,
        args.output_dim,
        args.max_tokens,
    ).eval()

    input_ids = torch.randint(low=0, high=model.config.text_config.vocab_size, size=(1, args.max_tokens), dtype=torch.int64)
    attention_mask = torch.ones((1, args.max_tokens), dtype=torch.int64)

    traced = torch.jit.trace(wrapper, (input_ids, attention_mask))
    output_path = args.output_dir / "qwen3_vl_text_fusion.pt"
    traced.save(str(output_path))
    print(f"Saved TorchScript text fusion encoder to {output_path}")


if __name__ == "__main__":
    main()
