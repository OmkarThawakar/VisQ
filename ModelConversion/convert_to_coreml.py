#!/usr/bin/env python3
"""Convert TorchScript assets into Core ML packages."""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import torch


def infer_tensor_spec(name: str, input_resolution: int, embedding_dim: int):
    if "vision" in name:
        if "qwen3_vl" in name:
            return [
                ct.TensorType(
                    name="pixel_values",
                    shape=(784, 1536),
                    dtype=float,
                ),
                ct.TensorType(
                    name="grid_thw",
                    shape=(1, 3),
                    dtype=int,
                ),
            ]
        return [
            ct.TensorType(
                name="pixel_values",
                shape=(1, 3, input_resolution, input_resolution),
                dtype=float,
            ),
            ct.TensorType(
                name="grid_thw",
                shape=(1, 3),
                dtype=int,
            )
            ]
    if "text_fusion" in name or "qwen3_vl_text" in name:
        return [
            ct.TensorType(
                name="input_ids",
                shape=(1, 32),
                dtype=int,
            ),
            ct.TensorType(
                name="attention_mask",
                shape=(1, 32),
                dtype=int,
            ),
        ]
    return [
        ct.TensorType(
            name="query_embedding",
            shape=(1, embedding_dim),
            dtype=float,
        )
    ]


def parse_compute_units(raw: str) -> ct.ComputeUnit:
    mapping = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }
    return mapping[raw.upper()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="vision")
    parser.add_argument("--onnx_path", type=Path)
    parser.add_argument("--torchscript_path", type=Path)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--input_resolution", type=int, default=448)
    parser.add_argument("--embedding_dim", type=int, default=1536)
    parser.add_argument("--compute_units", default="ALL")
    parser.add_argument("--quantize", default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torchscript_path = args.torchscript_path
    source_name = ""
    if torchscript_path is not None:
        source_name = torchscript_path.name.lower()
    elif args.onnx_path is not None:
        source_name = args.onnx_path.name.lower()
        torchscript_path = args.onnx_path.with_suffix(".pt")
    else:
        raise ValueError("Either --onnx_path or --torchscript_path must be provided")
    if not torchscript_path.exists():
        raise FileNotFoundError(f"Expected TorchScript file at {torchscript_path}")
    traced_model = torch.jit.load(str(torchscript_path))

    model = ct.convert(
        traced_model,
        source="pytorch",
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_units=parse_compute_units(args.compute_units),
        inputs=infer_tensor_spec(source_name, args.input_resolution, args.embedding_dim),
    )
    model.save(str(args.output_path))
    print(f"Saved Core ML package to {args.output_path}")


if __name__ == "__main__":
    main()
