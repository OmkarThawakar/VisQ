#!/usr/bin/env python3
"""
Patch existing .mlpackage files to use fixed input shapes instead of RangeDim.

Root cause of "ios18.add: Shapes are not compatible for broadcasting" on iPhone ANE:

  1. VisionEncoder: pixel_values is declared as RangeDim[64…4096] but the
     cached pos_embed constant is a fixed [784, D] tensor.  The ANE's shape
     propagation engine cannot prove RangeDim-N == 784 for the add op.

  2. Prefill: input_ids is declared as RangeDim[50…700] but visual_pos_mask is
     fixed [1, 285].  M-RoPE builds cos/sin of shape [1,1,285,head_dim] from
     the mask, while Q/K derive shape [1,heads,T_dynamic,head_dim] from
     embed_tokens(input_ids).  The ios18.add in q*cos + rotate_half(q)*sin
     fails because 285 ≠ T_dynamic in the ANE shape propagator.

Fix: change both inputs from flexible (RangeDim) to the exact static shapes
the models were TRACED with (N=784, T=285).  This does NOT require
re-tracing or re-converting — only the spec's FeatureDescription is updated;
the MIL program bytecode (and weights) are preserved.

Usage:
    pip install 'coremltools>=8.0'
    python3 scripts/patch_mlpackage_shapes.py
"""

import json
import os
import sys

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
COREML_DIR  = os.path.join(SCRIPT_DIR,
                            "../VisualSeek/VisualSeek/VisualSeek/CoreML")
CONFIG_PATH = os.path.join(COREML_DIR, "qwen2vl_config.json")

VE_PATH  = os.path.join(COREML_DIR, "Qwen2VLVisionEncoder.mlpackage")
PF_PATH  = os.path.join(COREML_DIR, "Qwen2VLPrefill.mlpackage")


# ── Read ground-truth dimensions from the saved config ──────────────────────

def read_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"WARNING: {CONFIG_PATH} not found — using hardcoded defaults")
        return {"patches_per_side": 28, "patch_flat_size": 1536,
                "prompt_length": 285}
    with open(CONFIG_PATH) as f:
        return json.load(f)


# ── Protobuf helpers ─────────────────────────────────────────────────────────

def _set_fixed_shape(array_feature_type, shape: list):
    """
    Replace any ShapeFlexibility (RangeDim / EnumeratedShapes) on an
    ArrayFeatureType proto with a plain fixed shape.
    """
    mat = array_feature_type
    # Clear whichever flexibility field is currently set
    flex_field = mat.WhichOneof("ShapeFlexibility")
    if flex_field:
        mat.ClearField(flex_field)
    # Write the fixed shape
    del mat.shape[:]
    mat.shape.extend(shape)


# ── Per-model patchers ───────────────────────────────────────────────────────

def patch_vision_encoder(path: str, raw_patches: int, patch_flat: int):
    """
    pixel_values: RangeDim[64…4096, 1536]  →  [raw_patches, patch_flat]
    """
    import coremltools as ct

    print(f"  Loading {os.path.basename(path)} …")
    model = ct.models.MLModel(path, skip_model_load=True)
    spec  = model.get_spec()

    patched = False
    for inp in spec.description.input:
        if inp.name == "pixel_values":
            _set_fixed_shape(inp.type.multiArrayType,
                             [raw_patches, patch_flat])
            print(f"  ✓  '{inp.name}'  →  [{raw_patches}, {patch_flat}]  (was flexible)")
            patched = True

    if not patched:
        print("  WARNING: 'pixel_values' input not found — skipping")
        return

    # get_spec() returns model._spec by reference — modifications are in-place.
    # No set_spec() needed; just save directly.
    model.save(path)
    print(f"  ✓  Saved → {path}")


def patch_prefill(path: str, prompt_len: int):
    """
    input_ids: RangeDim[50…700] (batch=1)  →  [1, prompt_len]
    """
    import coremltools as ct

    print(f"  Loading {os.path.basename(path)} …")
    model = ct.models.MLModel(path, skip_model_load=True)
    spec  = model.get_spec()

    patched = False
    for inp in spec.description.input:
        if inp.name == "input_ids":
            _set_fixed_shape(inp.type.multiArrayType, [1, prompt_len])
            print(f"  ✓  '{inp.name}'  →  [1, {prompt_len}]  (was flexible)")
            patched = True

    if not patched:
        print("  WARNING: 'input_ids' input not found — skipping")
        return

    # get_spec() returns model._spec by reference — modifications are in-place.
    model.save(path)
    print(f"  ✓  Saved → {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    try:
        import coremltools as ct
        print(f"coremltools {ct.__version__}")
    except ImportError:
        sys.exit("ERROR: pip install 'coremltools>=8.0'")

    cfg         = read_config()
    patches_ps  = cfg["patches_per_side"]          # 28
    raw_patches = patches_ps * patches_ps           # 784
    patch_flat  = cfg["patch_flat_size"]            # 1536
    prompt_len  = cfg["prompt_length"]              # 285

    print(f"\nConfig: raw_patches={raw_patches}, patch_flat={patch_flat}, "
          f"prompt_len={prompt_len}")

    for p, name in [(VE_PATH, "VisionEncoder"), (PF_PATH, "Prefill")]:
        if not os.path.exists(p):
            sys.exit(f"ERROR: {p} not found — run convert_qwen2vl_to_coreml.py first")

    print(f"\n[1/2] Patching VisionEncoder …")
    patch_vision_encoder(VE_PATH, raw_patches, patch_flat)

    print(f"\n[2/2] Patching Prefill …")
    patch_prefill(PF_PATH, prompt_len)

    print("\n" + "=" * 60)
    print("DONE.")
    print("The .mlpackage files have been patched in-place.")
    print("Clean-build the Xcode project and run on device.")
    print("=" * 60)


if __name__ == "__main__":
    main()
