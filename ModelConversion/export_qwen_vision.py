#!/usr/bin/env python3
"""Export lightweight placeholder ONNX models for local Core ML packaging.

These exports intentionally preserve the file names and input shapes that the
app expects, while using tiny deterministic PyTorch graphs instead of the full
Qwen weights. That lets the iOS project bundle real .mlpackage assets and keep
the current mock inference path until the production conversion pipeline exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


class VisionEncoderStub(torch.nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = pixel_values.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)
        embedding = pooled.repeat(1, self.embedding_dim)
        return torch.nn.functional.normalize(embedding, dim=-1)


class TextFusionStub(torch.nn.Module):
    def forward(self, query_embedding: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(query_embedding, dim=-1)


def export_models(output_dir: Path, input_resolution: int, embedding_dim: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    vision_model = VisionEncoderStub(embedding_dim).eval()
    vision_input = torch.rand(1, 3, input_resolution, input_resolution, dtype=torch.float32)
    vision_traced = torch.jit.trace(vision_model, vision_input)
    vision_traced.save(str(output_dir / "qwen_vision_encoder.pt"))
    torch.onnx.export(
        vision_model,
        vision_input,
        output_dir / "qwen_vision_encoder.onnx",
        input_names=["pixel_values"],
        output_names=["embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )

    text_fusion_model = TextFusionStub().eval()
    text_fusion_input = torch.rand(1, embedding_dim, dtype=torch.float32)
    text_fusion_traced = torch.jit.trace(text_fusion_model, text_fusion_input)
    text_fusion_traced.save(str(output_dir / "qwen_text_fusion.pt"))
    torch.onnx.export(
        text_fusion_model,
        text_fusion_input,
        output_dir / "qwen_text_fusion.onnx",
        input_names=["query_embedding"],
        output_names=["fused_embedding"],
        dynamic_axes={"query_embedding": {0: "batch"}, "fused_embedding": {0: "batch"}},
        opset_version=17,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--input_resolution", type=int, default=448)
    parser.add_argument("--embedding_dim", type=int, default=1536)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_models(args.output_dir, args.input_resolution, args.embedding_dim)
    print(f"Exported placeholder ONNX models to {args.output_dir}")


if __name__ == "__main__":
    main()
