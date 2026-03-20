#!/usr/bin/env python3
"""Load a Core ML package and print a small sanity summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--input_resolution", type=int, default=448)
    parser.add_argument("--embedding_dim", type=int, default=1536)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = ct.models.MLModel(str(args.model_path))
    spec = model.get_spec()
    input_name = spec.description.input[0].name

    if input_name == "pixel_values":
        sample = np.random.rand(1, 3, args.input_resolution, args.input_resolution).astype(np.float32)
    else:
        sample = np.random.rand(1, args.embedding_dim).astype(np.float32)

    prediction = model.predict({input_name: sample})
    output_name = next(iter(prediction))
    output = prediction[output_name]
    shape = getattr(output, "shape", None)
    print("Model loaded successfully")
    print(f"Input: {input_name}")
    print(f"Output: {output_name}")
    print(f"Output shape: {shape}")


if __name__ == "__main__":
    main()
