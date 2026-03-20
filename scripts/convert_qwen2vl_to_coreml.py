#!/usr/bin/env python3
"""
Convert Qwen3-VL-2B-Instruct to CoreML for iOS 17+ on-device deployment.

Produces four artefacts in --output-dir:
  Qwen2VLVisionEncoder.mlpackage   ~250 MB  (ViT + spatial merger + deepstack)
  Qwen2VLPrefill.mlpackage         ~800 MB  (LM prefill with image injection)
  Qwen2VLDecodeStep.mlpackage      ~800 MB  (single-token autoregressive step)
  qwen2vl_config.json              metadata the Swift runtime reads

Requirements (Python 3.10+):
  pip install torch transformers>=5.0 coremltools>=8.0 numpy einops accelerate torchvision

Usage:
  python3 scripts/convert_qwen2vl_to_coreml.py
  python3 scripts/convert_qwen2vl_to_coreml.py --quant int4   # recommended for device
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ─── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="Qwen/Qwen3-VL-2B-Instruct",
                   help="HuggingFace model ID")
    p.add_argument("--output-dir",  default="VisualSeek/VisualSeek/VisualSeek/CoreML",
                   help="Directory to write .mlpackage files")
    p.add_argument("--quant",       choices=["none", "fp16", "int8", "int4"],
                   default="int4", help="Weight quantization (int4 recommended for device)")
    p.add_argument("--max-seq-len", type=int, default=512,
                   help="Maximum sequence length for KV-cache allocation hint")
    p.add_argument("--image-size",  type=int, default=448,
                   help="Image resolution fed to the vision encoder")
    p.add_argument("--skip-done",   action="store_true",
                   help="Skip converting models whose .mlpackage already exists in output-dir")
    return p.parse_args()


# ─── Model loading ───────────────────────────────────────────────────────────

def load_model_and_processor(model_id: str):
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer
    except ImportError:
        sys.exit("ERROR: Install transformers>=5.0  →  pip install 'transformers>=5.0'")

    print(f"[1/4] Loading {model_id} on CPU in float16 (saves ~4 GB on 8 GB M2) …")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,   # float16 halves peak RSS (~4 GB vs ~8 GB)
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tc = model.config.text_config
    print(f"      hidden_size={tc.hidden_size}, "
          f"layers={tc.num_hidden_layers}, "
          f"kv_heads={tc.num_key_value_heads}, "
          f"head_dim={tc.head_dim}")
    return model, tokenizer


# ─── Image pre-processing helpers ────────────────────────────────────────────

def compute_image_constants(model_cfg, image_size: int):
    """Return the constants that are shared between Python export and Swift runtime."""
    vc = model_cfg.vision_config
    patch_size         = vc.patch_size            # 16
    temporal_ps        = vc.temporal_patch_size   # 2
    spatial_merge      = vc.spatial_merge_size    # 2

    patches_per_side   = image_size // patch_size                     # 28 for 448 px
    raw_patches        = patches_per_side * patches_per_side          # 784
    patch_flat_size    = 3 * temporal_ps * patch_size * patch_size    # 1536
    num_visual_tokens  = raw_patches // (spatial_merge ** 2)          # 196

    return {
        "patch_size":        patch_size,
        "temporal_ps":       temporal_ps,
        "spatial_merge":     spatial_merge,
        "patches_per_side":  patches_per_side,
        "raw_patches":       raw_patches,
        "patch_flat_size":   patch_flat_size,
        "num_visual_tokens": num_visual_tokens,
    }


# ─── Build prompt token IDs ──────────────────────────────────────────────────

SYSTEM_PROMPT = "You are a helpful assistant."
USER_PROMPT   = (
    "Describe this image in comprehensive detail for fine-grained image retrieval. "
    "Include: all visible people (count, appearance, clothing, activity, expression), "
    "all objects and their positions, scene type (indoor/outdoor), setting/venue, "
    "background elements, dominant colors, lighting conditions, time of day, "
    "composition, and any visible text."
)

IM_START_ID      = 151644   # <|im_start|>
IM_END_ID        = 151645   # <|im_end|>
VISION_START_ID  = 151652   # <|vision_start|>
VISION_END_ID    = 151653   # <|vision_end|>
IMAGE_PAD_ID     = 151655   # <|image_pad|>
NEWLINE_ID       = 198      # '\n'


def build_prompt_token_ids(tokenizer, num_visual_tokens: int):
    """
    Returns (token_id_list, image_start_index).
    image_start_index is the position of the FIRST <|image_pad|> token.
    """
    def tok(text):
        return tokenizer.encode(text, add_special_tokens=False)

    ids = []
    ids += [IM_START_ID] + tok("system") + [NEWLINE_ID]
    ids += tok(SYSTEM_PROMPT) + [IM_END_ID, NEWLINE_ID]
    ids += [IM_START_ID] + tok("user") + [NEWLINE_ID]
    ids += [VISION_START_ID]
    image_start_index = len(ids)           # ← position of first <|image_pad|>
    ids += [IMAGE_PAD_ID] * num_visual_tokens
    ids += [VISION_END_ID, NEWLINE_ID]
    ids += tok(USER_PROMPT) + [IM_END_ID, NEWLINE_ID]
    ids += [IM_START_ID] + tok("assistant") + [NEWLINE_ID]
    return ids, image_start_index


# ─── Patch vision attention for trace compatibility ───────────────────────────

def patch_pos_embed_for_tracing(visual_module, patches_per_side: int):
    """
    Replaces fast_pos_embed_interpolate with a version that:
    1. Pre-computes the FULL position embedding result as a float constant
    2. Returns it as a tensor addition (no indexing ops, no split ops)
    This avoids both torch.split(list_of_sizes) and int32/float32 matmul issues
    in coremltools.
    """
    import types as pytypes

    H = W = patches_per_side
    merge_size = visual_module.spatial_merge_size
    N = visual_module.num_grid_per_side

    # Pre-compute the full positional embedding for our fixed grid
    # using the ORIGINAL method (not traced, just eager)
    sample_thw = torch.tensor([[1, H, W]], dtype=torch.int32)
    with torch.no_grad():
        patch_embeds_const = visual_module.fast_pos_embed_interpolate(sample_thw)
        # patch_embeds_const shape: [T * H * W / merge_size^2 * merge_size^2, D]
        # = [H*W, D] = [784, D] for 448px

    # Detach and pin as a constant
    patch_embeds_const = patch_embeds_const.detach().clone()
    # Store on module so GC doesn't collect it
    visual_module._cached_pos_embeds = patch_embeds_const

    def patched_fast_pos_embed_interpolate(self, grid_thw):
        # Return the pre-computed constant — cast to match pos_embed dtype/device
        return self._cached_pos_embeds.to(
            device=self.pos_embed.weight.device,
            dtype=self.pos_embed.weight.dtype,
        )

    visual_module.fast_pos_embed_interpolate = pytypes.MethodType(
        patched_fast_pos_embed_interpolate, visual_module
    )


def patch_coremltools_int_op():
    """
    Fix coremltools 9.0 bug where `aten::Int` fails on 1-D single-element arrays.

    Root cause:
      `prim::NumToTensor` handler wraps the scalar val as ``[x.val]`` (list),
      creating a shape-(1,) MIL const instead of a shape-() scalar const.
      The downstream ``_int`` / ``_cast`` handler then does ``int(x.val)`` where
      ``x.val = np.array([16])`` — raises
      "TypeError: only 0-dimensional arrays can be converted to Python scalars".

    Fix: patch ``_cast`` to flatten the value before the Python ``int()`` call.
    """
    try:
        import coremltools.converters.mil.frontend.torch.ops as ct_ops
        import numpy as _np

        # Grab the original _cast; we'll wrap it
        _orig_cast = ct_ops._cast

        def _patched_cast(context, node, dtype, dtype_name):
            from coremltools.converters.mil.mil import Builder as mb
            inputs = ct_ops._get_inputs(context, node, expected=1)
            x = inputs[0]

            if not (_np.all([d == 1 for d in x.shape]) or len(x.shape) == 0):
                raise ValueError(
                    f"input to cast must be either a scalar or a length 1 tensor, got shape {x.shape}"
                )

            if x.can_be_folded_to_const():
                if not isinstance(x.val, dtype):
                    # Use .flat[0] or item() to handle both 0-D and (1,)-shaped arrays
                    raw = _np.asarray(x.val).flat[0]
                    res = mb.const(val=dtype(raw), name=node.name)
                else:
                    res = x
            elif len(x.shape) > 0:
                x = mb.squeeze(x=x, name=node.name + "_item")
                res = mb.cast(x=x, dtype=dtype_name, name=node.name)
            else:
                res = mb.cast(x=x, dtype=dtype_name, name=node.name)
            context.add(res, node.name)

        ct_ops._cast = _patched_cast
        print("  Patched coremltools _cast to handle 1-D scalar arrays (aten::Int fix)")
    except Exception as exc:
        print(f"  WARNING: Could not patch coremltools _cast: {exc}")


def force_eager_attention_for_tracing(model):
    """
    Switches the text model's attention implementation to 'eager' mode.

    coremltools 9.0 has a bug in its SDPA (scaled_dot_product_attention) handler:
    ``_cast_bool_attn_mask`` casts bool → fp16 then does ``mb.sub(x=1.0, y=fp16)``
    where ``1.0`` is float32 — a dtype mismatch that crashes conversion.

    Eager attention uses explicit matmul + softmax, which coremltools handles
    correctly. Performance on Neural Engine is equivalent for our batch-1 case.
    """
    try:
        model.config._attn_implementation = "eager"
        model.config.text_config._attn_implementation = "eager"
        # Also update each text layer's config
        for layer in model.model.language_model.layers:
            layer.self_attn.config._attn_implementation = "eager"
        print("  Switched text model to eager attention (avoids SDPA coremltools bug)")
    except Exception as exc:
        print(f"  WARNING: Could not switch to eager attention: {exc}")


def patch_text_rotate_half_for_tracing(text_head_dim: int):
    """
    Monkey-patches the ``rotate_half`` function in the qwen3_vl modeling module
    to use a static Python-int half-size captured in a closure, instead of the
    dynamic ``x.shape[-1] // 2`` expression.

    ``x.shape[-1] // 2`` generates:
        aten::size(x, -1)          → int 128
        prim::NumToTensor(128)     → scalar Tensor(128)
        aten::floor_divide(T, 2)   → scalar Tensor(64)
        aten::Int(T)               → int 64

    coremltools 9.0 fails on ``aten::Int`` of the result because ``x.val``
    ends up as a non-0-d numpy array when the value flows through
    ``prim::NumToTensor`` + ``floor_divide``.

    Using a static Python ``int`` in the closure makes the slice indices
    literal constants in the TorchScript graph, completely avoiding the chain.
    """
    half = text_head_dim // 2  # e.g. 64 for head_dim=128 — Python int constant

    def _trace_rotate_half(x):
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    try:
        import transformers.models.qwen3_vl.modeling_qwen3_vl as _mod
        _mod.rotate_half = _trace_rotate_half
        # Also patch apply_rotary_pos_emb_vision's internal rotate_half reference
        # by replacing the function that apply_rotary_pos_emb calls via closure
        print(f"  Patched qwen3_vl.rotate_half with static half_dim={half}")
    except Exception as e:
        print(f"  WARNING: Could not patch rotate_half: {e}")


def patch_interleaved_mrope_for_tracing(model, mrope_section: list, head_dim: int):
    """
    Patches Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope to avoid
    strided in-place __setitem__ (slice step != 1), which coremltools 9.0
    translates to a broken slice_update with wrong slice shape.

    Original code:
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)            # stride-3 slice
            freqs_t[..., idx] = freqs[dim, ..., idx]  # in-place strided write

    Fix: replace with torch.where using CONSTANT boolean masks computed once
    outside the trace. torch.where(const_mask, a, b) is a pure function that
    coremltools represents as ios18.where — no slice_update needed.
    """
    import types as pytypes

    half_dim  = head_dim // 2            # 64
    H_section = mrope_section[1]         # 20
    W_section = mrope_section[2]         # 20
    length_H  = H_section * 3            # 60 — upper bound of strided slice
    length_W  = W_section * 3            # 60

    # Precompute static boolean masks as Python-side constants.
    # Shape (half_dim,) — broadcast over (batch, seq_len) during trace.
    H_mask = torch.zeros(half_dim, dtype=torch.bool)
    W_mask = torch.zeros(half_dim, dtype=torch.bool)
    H_mask[list(range(1, length_H, 3))] = True   # positions 1,4,7,...,58
    W_mask[list(range(2, length_W, 3))] = True   # positions 2,5,8,...,59

    def _patched_interleaved_mrope(self_re, freqs, mrope_section_arg):
        # freqs: (3, bs, seq_len, half_dim)
        # Use pre-computed constant masks — fully static, no __setitem__.
        result = torch.where(H_mask.to(freqs.device), freqs[1], freqs[0])
        result = torch.where(W_mask.to(freqs.device), freqs[2], result)
        return result

    try:
        rotary_emb = model.model.language_model.rotary_emb
        rotary_emb.apply_interleaved_mrope = pytypes.MethodType(
            _patched_interleaved_mrope, rotary_emb
        )
        print(f"  Patched rotary_emb.apply_interleaved_mrope with torch.where "
              f"(half_dim={half_dim}, H_mask[{H_section}], W_mask[{W_section}])")
    except Exception as e:
        print(f"  WARNING: Could not patch apply_interleaved_mrope: {e}")


def patch_rot_pos_emb_for_tracing(visual_module, patches_per_side: int):
    """
    Pre-computes the rotary position embedding output as a constant for our
    fixed image grid.

    `rot_pos_emb(grid_thw)` calls `self.rotary_pos_emb(max_grid_size)` where
    `max_grid_size` is a traced tensor derived from `grid_thw`.  Inside that
    sub-module, `torch.outer(arange(max_grid_size), inv_freq)` is called.
    coremltools 9.0 fails because `arange` of a traced tensor produces int64
    while `inv_freq` is float32 → matmul/outer x=int64, y=fp32 error.

    Fix: eagerly compute the output for our fixed grid once and return it as
    a constant tensor during tracing.
    """
    import types as pytypes

    H = W = patches_per_side
    sample_thw = torch.tensor([[1, H, W]], dtype=torch.int32)

    with torch.no_grad():
        rot_emb_const = visual_module.rot_pos_emb(sample_thw).detach().clone()

    visual_module._cached_rot_pos_emb = rot_emb_const

    def patched_rot_pos_emb(self, grid_thw):
        return self._cached_rot_pos_emb

    visual_module.rot_pos_emb = pytypes.MethodType(patched_rot_pos_emb, visual_module)


def patch_vision_attention_for_tracing(visual_module):
    """
    Replaces each ViT block's attention forward with a simple full-sequence
    standard-attention forward that avoids torch.split(lengths.tolist(), …).
    This is safe because we always run a single image (no batch packing).

    Key trace-compatibility fixes:
    - Use explicit head_dim (not -1) in reshape → avoids dynamic aten::Int ops
    - Capture head_dim//2 as a Python constant in the closure → _rotate_half
      never generates aten::floor_divide / aten::Int ops
    """
    def make_patched_attn_forward(orig_attn):
        # Capture static integers from the attention module as Python constants
        # so they never become dynamic tensor operations in the trace graph.
        num_heads = int(orig_attn.num_heads)   # Python int, not a tensor attribute
        head_dim  = int(orig_attn.head_dim)    # Python int
        half_dim  = head_dim // 2              # Python int — avoids aten::Int op
        embed_dim = num_heads * head_dim       # Python int for final reshape

        def _rotate_half(x):
            # Use static half_dim (Python int) rather than x.shape[-1]//2 (dynamic)
            return torch.cat([-x[..., half_dim:], x[..., :half_dim]], dim=-1)

        def patched_forward(hidden_states, cu_seqlens=None, rotary_pos_emb=None,
                            position_embeddings=None, **kwargs):
            seq_len = hidden_states.shape[0]
            qkv = orig_attn.qkv(hidden_states)
            # Explicit head_dim instead of -1 avoids a dynamic shape computation
            # that later generates aten::floor_divide / aten::Int in the trace.
            q, k, v = (qkv
                       .reshape(seq_len, 3, num_heads, head_dim)
                       .permute(1, 0, 2, 3)
                       .unbind(0))

            if position_embeddings is not None:
                cos, sin = position_embeddings   # [seq_len, head_dim*2] (cat'd)
                # Unsqueeze to [seq_len, 1, head_dim*2] for broadcast over heads
                cos2 = cos.unsqueeze(-2).float()
                sin2 = sin.unsqueeze(-2).float()
                q = (q.float() * cos2 + _rotate_half(q.float()) * sin2).to(q.dtype)
                k = (k.float() * cos2 + _rotate_half(k.float()) * sin2).to(k.dtype)

            # Standard full-sequence attention — [1, num_heads, seq_len, head_dim]
            q = q.unsqueeze(0).transpose(1, 2)
            k = k.unsqueeze(0).transpose(1, 2)
            v = v.unsqueeze(0).transpose(1, 2)

            attn = torch.matmul(q, k.transpose(-2, -1)) * orig_attn.scaling
            attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)
            out = torch.matmul(attn, v)           # [1, H, S, D]
            out = out.squeeze(0).transpose(0, 1)  # [S, H, D]
            # Use static embed_dim instead of -1 to avoid dynamic shape inference
            out = out.reshape(seq_len, embed_dim).contiguous()
            return orig_attn.proj(out)
        return patched_forward

    for block in visual_module.blocks:
        block.attn.forward = make_patched_attn_forward(block.attn)


def patch_deepstack_for_tracing(language_model, image_start_idx: int, n_vis: int):
    """
    Patches the language model's _deepstack_process to use static slice indexing
    instead of bool-masked scatter/index_put_.

    Original code (not trace-compatible):
        visual_hidden = hidden_states[visual_pos_masks]   # bool fancy-index
        updated = visual_hidden + visual_embeds
        hidden_states[visual_pos_masks] = updated         # index_put_ with bool

    coremltools 9.0 cannot represent index_put_ with a bool mask whose shape
    doesn't statically match the input tensor shape (fails with:
      "index shape (1,T) must match input shape (1,is52,H)").

    Fix: replace with an explicit slice that uses the known Python-int positions
    [image_start_idx : image_start_idx + n_vis] — completely static, no fancy
    indexing ops in the trace graph.
    """
    import types as pytypes

    s = image_start_idx
    e = image_start_idx + n_vis

    # Actual call signature (from modeling_qwen3_vl.py line 933):
    #   self._deepstack_process(hidden_states, visual_pos_masks, visual_embeds)
    def patched_deepstack_process(self_lm, hidden_states, visual_pos_masks, visual_embeds):
        # hidden_states: [1, T, H] (3-D batch tensor)
        # visual_embeds: [V, H] — expand to [1, V, H] for slice broadcast
        ve = visual_embeds if visual_embeds.dim() == 3 else visual_embeds.unsqueeze(0)
        # clone() prevents in-place mutation on leaf tensors
        hidden_states = hidden_states.clone()
        hidden_states[:, s:e, :] = (
            hidden_states[:, s:e, :] + ve
        ).to(hidden_states.dtype)
        return hidden_states

    patched = False
    if hasattr(language_model, '_deepstack_process'):
        language_model._deepstack_process = pytypes.MethodType(
            patched_deepstack_process, language_model
        )
        print(f"  Patched language_model._deepstack_process → slice [{s}:{e}]")
        patched = True
    if not patched:
        print("  WARNING: language_model has no _deepstack_process — "
              "bool-index scatter may still fail during prefill conversion")


# ─── Wrapper: Vision Encoder ─────────────────────────────────────────────────

class VisionEncoderWrapper(nn.Module):
    """
    Qwen3-VL vision encoder wrapper.
    Input:  pixel_values [N_patches, patch_flat_size], grid_thw [1, 3]
    Output: (pooler_output [V, H], ds0 [V, H], ds1 [V, H], ds2 [V, H])
            where V = num_visual_tokens, H = hidden_size
    """
    def __init__(self, visual_module):
        super().__init__()
        self.visual = visual_module

    def forward(self, pixel_values, grid_thw):
        out = self.visual(pixel_values, grid_thw)
        # transformers >= 5.x returns (pooler_output, deepstack_list) as a plain tuple
        if isinstance(out, tuple):
            pooler_output = out[0]
            deepstack = out[1]
        else:
            pooler_output = out.pooler_output
            deepstack = out.deepstack_features
        return (pooler_output, deepstack[0], deepstack[1], deepstack[2])


# ─── Wrapper: Prefill ────────────────────────────────────────────────────────

class PrefillWrapper(nn.Module):
    """
    Embeds the full prompt, injects image features at the fixed slice,
    injects deepstack features, runs the LM in prefill mode, and returns:
      logits   [1, vocab_size]
      kv_keys  [L, 1, kv_heads, T, head_dim]
      kv_vals  [L, 1, kv_heads, T, head_dim]

    visual_pos_mask is a bool tensor [1, T] — True at the V image-pad positions.
    """
    def __init__(self, lm, lm_head, config, n_layers, image_start_idx, n_vis):
        super().__init__()
        self.lm              = lm
        self.lm_head         = lm_head
        self.config          = config
        self.n_layers        = n_layers
        self.image_start_idx = image_start_idx
        self.n_vis           = n_vis

    def forward(self, input_ids, image_features, ds0, ds1, ds2, visual_pos_mask):
        from transformers.cache_utils import DynamicCache

        embeds = self.lm.embed_tokens(input_ids)   # [1, T, H]
        s = self.image_start_idx
        e = s + self.n_vis
        # Use in-place slice assignment instead of torch.cat.
        # torch.cat makes the sequence dimension symbolic (is52) in coremltools
        # because it cannot prove the output shape equals the input shape.
        # With slice assignment the tensor shape stays static (T is a Python int
        # known at trace time), so downstream bool-indexed ops see [1, T, H].
        embeds = embeds.clone()
        embeds[:, s:e, :] = image_features.unsqueeze(0).to(embeds.dtype)

        cache = DynamicCache()
        out = self.lm(
            inputs_embeds=embeds,
            past_key_values=cache,
            use_cache=True,
            visual_pos_masks=visual_pos_mask,
            deepstack_visual_embeds=[ds0, ds1, ds2],
        )

        # transformers 5.3.0: DynamicCache uses .layers[i].keys / .layers[i].values
        new_k = torch.stack([out.past_key_values.layers[i].keys   for i in range(self.n_layers)])
        new_v = torch.stack([out.past_key_values.layers[i].values for i in range(self.n_layers)])
        logits = self.lm_head(out.last_hidden_state[:, -1, :])
        return logits, new_k, new_v


# ─── Wrapper: Single decode step ─────────────────────────────────────────────

class DecodeStepWrapper(nn.Module):
    """
    One autoregressive step (no deepstack — only used during prefill).
    Inputs:
      token_id [1, 1]                          int64
      kv_keys  [L, 1, kv_heads, past_len, H]  float16
      kv_vals  [L, 1, kv_heads, past_len, H]  float16
    Outputs:
      logits       [1, vocab_size]
      new_kv_keys  [L, 1, kv_heads, past_len+1, H]
      new_kv_vals  [L, 1, kv_heads, past_len+1, H]
    """
    def __init__(self, lm, lm_head, config, n_layers):
        super().__init__()
        self.lm       = lm
        self.lm_head  = lm_head
        self.config   = config
        self.n_layers = n_layers

    def forward(self, token_id, kv_keys, kv_vals):
        from transformers.cache_utils import DynamicCache, DynamicLayer

        embeds = self.lm.embed_tokens(token_id)   # [1, 1, H]

        # Pre-populate a DynamicCache from the stacked prefill KV tensors.
        # transformers 5.3.0: DynamicCache stores per-layer DynamicLayer objects
        # with .keys / .values tensors of shape [1, kv_h, past_len, D].
        cache = DynamicCache()
        for i in range(self.n_layers):
            layer = DynamicLayer()
            layer.lazy_initialization(kv_keys[i], kv_vals[i])  # sets dtype/device, marks initialized
            layer.keys   = kv_keys[i]   # shape [1, kv_h, past_len, D]
            layer.values = kv_vals[i]
            cache.layers.append(layer)

        out = self.lm(inputs_embeds=embeds, past_key_values=cache, use_cache=True)

        # After forward, cache has one new token appended per layer
        new_k = torch.stack([out.past_key_values.layers[i].keys   for i in range(self.n_layers)])
        new_v = torch.stack([out.past_key_values.layers[i].values for i in range(self.n_layers)])
        logits = self.lm_head(out.last_hidden_state[:, -1, :])
        return logits, new_k, new_v


# ─── CoreML conversion helpers ───────────────────────────────────────────────

def _apply_quantization(mlmodel, quant: str):
    if quant == "none":
        return mlmodel
    try:
        import coremltools.optimize.coreml as cto
    except ImportError:
        print("  ⚠  coremltools.optimize not found — skipping quantization")
        return mlmodel

    if quant in ("int4", "int8"):
        nbits = 4 if quant == "int4" else 8
        cfg = cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype=f"int{nbits}",
                granularity="per_block",
                block_size=32,
            )
        )
        print(f"  Applying {quant} weight quantization …")
        mlmodel = cto.linear_quantize_weights(mlmodel, config=cfg)
    return mlmodel


def _convert(wrapper, sample_inputs, input_specs, output_names, min_target, quant, label,
             compute_precision=None, compute_units=None):
    import coremltools as ct

    print(f"\n[converting] {label}")

    if compute_precision is None:
        compute_precision = ct.precision.FLOAT16
    if compute_units is None:
        compute_units = ct.ComputeUnit.ALL

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, sample_inputs, strict=False)
        traced.eval()

    mlmodel = ct.convert(
        traced,
        inputs=input_specs,
        outputs=[ct.TensorType(name=n) for n in output_names],
        minimum_deployment_target=min_target,
        compute_precision=compute_precision,
        compute_units=compute_units,
        convert_to="mlprogram",
        # skip_model_load avoids coremltools trying to compile/validate the model
        # on this macOS dev machine (the MPS/ANE compiler crashes for some models).
        # The .mlpackage can be deployed and compiled on-device (iOS) without issue.
        skip_model_load=True,
    )
    mlmodel = _apply_quantization(mlmodel, quant)
    return mlmodel


# ─── Export: Vision encoder ──────────────────────────────────────────────────

def export_vision_encoder(model, img_const, output_dir, quant):
    import coremltools as ct

    # Convert vision module to float32 BEFORE wrapping/tracing.
    # coremltools 9.0 layer_norm validation requires epsilon (always fp32)
    # and gamma to have the same dtype.  When the model is loaded in fp16
    # the gamma weights are fp16, causing:
    #   "epsilon has dtype fp32 whereas gamma has dtype fp16"
    # Converting to float32 makes gamma fp32 ≡ epsilon fp32.  The vision
    # encoder is ~250 MB in fp32, well within the 8 GB M2 budget.
    print("  Converting vision encoder weights to float32 (fixes LayerNorm dtype mismatch) …")
    model.model.visual.float()

    wrapper = VisionEncoderWrapper(model.model.visual)
    wrapper.eval()

    N    = img_const["raw_patches"]      # 784 for 448-px
    flat = img_const["patch_flat_size"]  # 1536

    # Use float32 sample inputs to match the now-float32 visual module
    sample_pv  = torch.randn(N, flat, dtype=torch.float32)
    sample_thw = torch.tensor([[1, img_const["patches_per_side"],
                                   img_const["patches_per_side"]]], dtype=torch.long)

    # Use fixed shapes (not RangeDim) so the ANE can propagate shapes through
    # the cached pos_embed constant (shape [N, D]).  A RangeDim here creates a
    # symbolic N that the ANE cannot reconcile with the fixed [N, D] constant,
    # causing "ios18.add: Shapes are not compatible for broadcasting" at runtime.
    specs = [
        ct.TensorType(name="pixel_values", shape=[N, flat], dtype=np.float32),
        ct.TensorType(name="grid_thw",     shape=[1, 3],    dtype=np.int32),
    ]

    # Use FLOAT32 compute precision for the vision encoder (weights are float32,
    # and coremltools will quantize them via _apply_quantization if requested).
    mlmodel = _convert(wrapper, (sample_pv, sample_thw),
                       specs,
                       ["pooler_output", "deepstack_0", "deepstack_1", "deepstack_2"],
                       ct.target.iOS18, quant,
                       "Qwen2VLVisionEncoder",
                       compute_precision=ct.precision.FLOAT32)

    path = os.path.join(output_dir, "Qwen2VLVisionEncoder.mlpackage")
    mlmodel.save(path)
    print(f"  ✓  Saved → {path}")


# ─── Export: Prefill ─────────────────────────────────────────────────────────

def export_prefill(model, img_const, prompt_ids, image_start_index, output_dir, quant):
    import coremltools as ct

    tc     = model.config.text_config
    L      = tc.num_hidden_layers
    kv_h   = tc.num_key_value_heads
    h_dim  = tc.head_dim
    hidden = tc.hidden_size
    V      = img_const["num_visual_tokens"]   # 196
    T      = len(prompt_ids)

    wrapper = PrefillWrapper(
        lm               = model.model.language_model,
        lm_head          = model.lm_head,
        config           = model.config,
        n_layers         = L,
        image_start_idx  = image_start_index,
        n_vis            = V,
    )
    wrapper.eval()

    sample_ids  = torch.tensor([prompt_ids], dtype=torch.long)     # [1, T]
    sample_feat = torch.randn(V, hidden, dtype=torch.float16)       # [V, H]
    sample_ds   = torch.randn(V, hidden, dtype=torch.float16)       # [V, H]
    sample_mask = torch.zeros(1, T, dtype=torch.bool)
    sample_mask[0, image_start_index:image_start_index+V] = True

    # input_ids must be fixed shape [1, T] (not RangeDim) because visual_pos_mask
    # is [1, T] (fixed) and the LM derives M-RoPE position_ids from it.
    # A RangeDim on input_ids creates a T_dynamic that conflicts with T_fixed in
    # the M-RoPE cos/sin tensors, causing "ios18.add: Shapes are not compatible
    # for broadcasting" when the ANE propagates shapes through q*cos + rotate_half(q)*sin.
    specs = [
        ct.TensorType(name="input_ids",       shape=[1, T],      dtype=np.int32),
        ct.TensorType(name="image_features",  shape=[V, hidden], dtype=np.float32),
        ct.TensorType(name="deepstack_0",     shape=[V, hidden], dtype=np.float32),
        ct.TensorType(name="deepstack_1",     shape=[V, hidden], dtype=np.float32),
        ct.TensorType(name="deepstack_2",     shape=[V, hidden], dtype=np.float32),
        ct.TensorType(name="visual_pos_mask", shape=[1, T],      dtype=np.bool_),
    ]

    mlmodel = _convert(wrapper,
                       (sample_ids, sample_feat, sample_ds, sample_ds, sample_ds, sample_mask),
                       specs,
                       ["logits", "kv_keys", "kv_vals"],
                       ct.target.iOS18, quant,
                       "Qwen2VLPrefill")

    path = os.path.join(output_dir, "Qwen2VLPrefill.mlpackage")
    mlmodel.save(path)
    print(f"  ✓  Saved → {path}")


# ─── Export: Decode step ─────────────────────────────────────────────────────

def export_decode_step(model, img_const, prompt_len, output_dir, quant):
    import coremltools as ct

    tc    = model.config.text_config
    L     = tc.num_hidden_layers
    kv_h  = tc.num_key_value_heads
    h_dim = tc.head_dim

    wrapper = DecodeStepWrapper(
        lm       = model.model.language_model,
        lm_head  = model.lm_head,
        config   = model.config,
        n_layers = L,
    )
    wrapper.eval()

    S = prompt_len   # initial past_len for tracing
    sample_tok  = torch.tensor([[0]], dtype=torch.long)
    sample_keys = torch.randn(L, 1, kv_h, S, h_dim, dtype=torch.float16)
    sample_vals = torch.randn(L, 1, kv_h, S, h_dim, dtype=torch.float16)

    specs = [
        ct.TensorType(name="token_id", shape=[1, 1], dtype=np.int32),
        ct.TensorType(
            name="kv_keys",
            shape=ct.Shape(shape=(
                L, 1, kv_h,
                ct.RangeDim(lower_bound=1, upper_bound=700, default=S),
                h_dim,
            )),
            dtype=np.float32,
        ),
        ct.TensorType(
            name="kv_vals",
            shape=ct.Shape(shape=(
                L, 1, kv_h,
                ct.RangeDim(lower_bound=1, upper_bound=700, default=S),
                h_dim,
            )),
            dtype=np.float32,
        ),
    ]

    mlmodel = _convert(wrapper, (sample_tok, sample_keys, sample_vals),
                       specs,
                       ["logits", "new_kv_keys", "new_kv_vals"],
                       ct.target.iOS18, quant,
                       "Qwen2VLDecodeStep")

    path = os.path.join(output_dir, "Qwen2VLDecodeStep.mlpackage")
    mlmodel.save(path)
    print(f"  ✓  Saved → {path}")


# ─── Write config ─────────────────────────────────────────────────────────────

def write_config(model, img_const, image_start_index, prompt_len, output_dir):
    tc = model.config.text_config
    config = {
        "model_id":           "Qwen/Qwen3-VL-2B-Instruct",
        "num_visual_tokens":  img_const["num_visual_tokens"],
        "patch_flat_size":    img_const["patch_flat_size"],
        "patches_per_side":   img_const["patches_per_side"],
        "image_start_index":  image_start_index,
        "prompt_length":      prompt_len,
        "max_new_tokens":     200,
        "hidden_size":        tc.hidden_size,
        "num_layers":         tc.num_hidden_layers,
        "num_kv_heads":       tc.num_key_value_heads,
        "head_dim":           tc.head_dim,
        "vocab_size":         tc.vocab_size,
        "eos_token_id":       IM_END_ID,
        "image_pad_id":       IMAGE_PAD_ID,
        "vision_start_id":    VISION_START_ID,
        "vision_end_id":      VISION_END_ID,
        "im_start_id":        IM_START_ID,
        "im_end_id":          IM_END_ID,
        "num_deepstack":      3,
        "kv_cache_format":    "separate_keys_vals",
    }
    path = os.path.join(output_dir, "qwen2vl_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓  Config → {path}")
    return config


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    try:
        import coremltools as ct
        print(f"coremltools {ct.__version__}")
    except ImportError:
        sys.exit("ERROR: pip install 'coremltools>=8.0'")

    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model_and_processor(args.model)

    img_const = compute_image_constants(model.config, args.image_size)
    print(f"[image] {args.image_size}×{args.image_size} → "
          f"{img_const['raw_patches']} patches → "
          f"{img_const['num_visual_tokens']} visual tokens")

    prompt_ids, image_start_index = build_prompt_token_ids(
        tokenizer, img_const["num_visual_tokens"]
    )
    print(f"[prompt] {len(prompt_ids)} tokens, image starts at index {image_start_index}")

    print("\nPatching modules and coremltools for trace compatibility …")
    # Fix coremltools 9.0 bug: aten::Int fails on 1-D scalar arrays from NumToTensor
    patch_coremltools_int_op()
    # Switch text attention to eager (avoids coremltools SDPA bool-mask fp32/fp16 bug)
    force_eager_attention_for_tracing(model)
    # Patch rotate_half BEFORE any tracing — applies to both vision and text models.
    # Qwen3-VL uses M-RoPE: rotate_half is called on q_pe/k_pe which have size
    # rope_dim = sum(mrope_section) = 24+20+20 = 64, NOT head_dim=128.
    # Correct half = rope_dim // 2 = 32.  Using head_dim//2=64 is wrong and causes
    # 'mps.strided_slice_update update[2]=20 vs shape(data)[2]=64' crash.
    rope_scaling  = getattr(model.config.text_config, "rope_scaling", None) or {}
    if not isinstance(rope_scaling, dict):
        rope_scaling = vars(rope_scaling) if hasattr(rope_scaling, "__dict__") else {}
    mrope_section = rope_scaling.get("mrope_section", [])
    rope_dim = sum(mrope_section) if mrope_section else model.config.text_config.head_dim
    patch_text_rotate_half_for_tracing(rope_dim)   # half = rope_dim // 2 = 32
    patch_interleaved_mrope_for_tracing(model, mrope_section, model.config.text_config.head_dim)
    patch_rot_pos_emb_for_tracing(model.model.visual, img_const["patches_per_side"])
    patch_pos_embed_for_tracing(model.model.visual, img_const["patches_per_side"])
    patch_vision_attention_for_tracing(model.model.visual)
    # Patch deepstack before prefill export — replaces bool-indexed scatter with
    # a static slice so coremltools can represent it in the MIL graph.
    patch_deepstack_for_tracing(
        model.model.language_model,
        image_start_idx=image_start_index,
        n_vis=img_const["num_visual_tokens"],
    )

    def _done(name):
        path = os.path.join(args.output_dir, name)
        if args.skip_done and os.path.exists(path):
            print(f"  (skipping — {path} already exists)")
            return True
        return False

    print("\n[2/4] Exporting Vision Encoder …")
    if not _done("Qwen2VLVisionEncoder.mlpackage"):
        export_vision_encoder(model, img_const, args.output_dir, args.quant)
        gc.collect()

    print("\n[3/4] Exporting Prefill model …")
    if not _done("Qwen2VLPrefill.mlpackage"):
        export_prefill(model, img_const, prompt_ids, image_start_index,
                       args.output_dir, args.quant)
        gc.collect()

    print("\n[4/4] Exporting Decode Step model …")
    if not _done("Qwen2VLDecodeStep.mlpackage"):
        export_decode_step(model, img_const, len(prompt_ids),
                           args.output_dir, args.quant)
        gc.collect()

    print("\n[config] Writing runtime config …")
    cfg = write_config(model, img_const, image_start_index,
                       len(prompt_ids), args.output_dir)

    print("\n" + "="*60)
    print("DONE. Copy these files into Xcode:")
    for name in ["Qwen2VLVisionEncoder.mlpackage",
                 "Qwen2VLPrefill.mlpackage",
                 "Qwen2VLDecodeStep.mlpackage",
                 "qwen2vl_config.json"]:
        print(f"  {os.path.join(args.output_dir, name)}")
    print(f"\nimage_start_index = {image_start_index}")
    print(f"num_visual_tokens = {img_const['num_visual_tokens']}")
    print(f"kv_heads = {model.config.text_config.num_key_value_heads}")
    print("="*60)


if __name__ == "__main__":
    main()
