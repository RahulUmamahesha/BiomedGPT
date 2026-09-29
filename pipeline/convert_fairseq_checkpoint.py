"""
Converts a fairseq-format OFA/BiomedGPT checkpoint (.pt) into this project's
HF-transformers checkpoint layout by renaming state_dict keys (see RENAME_RULES)
and dropping fairseq-only bookkeeping buffers with no HF counterpart.
"""
import argparse
import os
import re
import shutil

import torch
from transformers import OFAModel

RENAME_RULES = [
    (re.compile(r"^decoder\.layers\.(\d+)\.encoder_attn_layer_norm\."), r"decoder.layers.\1.cross_attn_layer_norm."),
    (re.compile(r"^decoder\.layers\.(\d+)\.encoder_attn\."), r"decoder.layers.\1.cross_attn."),
    (re.compile(r"^decoder\.layers\.(\d+)\.cross_attn_ln\."), r"decoder.layers.\1.cross_attn_mid_layer_norm."),
    (re.compile(r"^decoder\.layers\.(\d+)\.self_attn_ln\."), r"decoder.layers.\1.self_attn_mid_layer_norm."),
    (re.compile(r"^decoder\.layers\.(\d+)\.ffn_layernorm\."), r"decoder.layers.\1.ffn_layer_norm."),
    (re.compile(r"^encoder\.layers\.(\d+)\.attn_ln\."), r"encoder.layers.\1.self_attn_mid_layer_norm."),
    (re.compile(r"^encoder\.layers\.(\d+)\.ffn_layernorm\."), r"encoder.layers.\1.ffn_layer_norm."),
]
DROP_KEYS = {"encoder.version", "decoder.version"}


def convert_state_dict(fs_sd):
    converted = {}
    for key, tensor in fs_sd.items():
        if key in DROP_KEYS:
            continue
        new_key = key
        for pattern, replacement in RENAME_RULES:
            new_key = pattern.sub(replacement, new_key)
        converted[new_key] = tensor
    return converted


def convert(fairseq_ckpt_path, template_hf_dir, output_dir):
    print(f"Loading fairseq checkpoint from {fairseq_ckpt_path} (safe weights_only mode)...")
    torch.serialization.add_safe_globals([argparse.Namespace])
    ckpt = torch.load(fairseq_ckpt_path, map_location="cpu", weights_only=True)
    fs_sd = ckpt["model"]

    converted_sd = convert_state_dict(fs_sd)

    print("Validating against template HF model...")
    template_model = OFAModel.from_pretrained(template_hf_dir, use_cache=False)
    template_keys = set(template_model.state_dict().keys())
    converted_keys = set(converted_sd.keys())

    missing = template_keys - converted_keys
    unexpected = converted_keys - template_keys
    if missing or unexpected:
        raise RuntimeError(f"Key mismatch after conversion. Missing: {missing}\nUnexpected: {unexpected}")

    shape_mismatches = [
        k for k in template_keys
        if tuple(converted_sd[k].shape) != tuple(template_model.state_dict()[k].shape)
    ]
    if shape_mismatches:
        raise RuntimeError(f"Shape mismatches after conversion: {shape_mismatches}")

    print("Key/shape validation passed. Loading converted weights into model to double check...")
    template_model.load_state_dict(converted_sd, strict=True)
    print("load_state_dict succeeded with strict=True.")

    os.makedirs(output_dir, exist_ok=True)
    for fname in ["config.json", "vocab.json", "merges.txt"]:
        shutil.copy(os.path.join(template_hf_dir, fname), os.path.join(output_dir, fname))

    out_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(converted_sd, out_path)
    print(f"Wrote converted checkpoint to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fairseq-ckpt", required=True)
    parser.add_argument("--template-hf-dir", default="checkpoints/instruct-biomedgpt-base")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    convert(args.fairseq_ckpt, args.template_hf_dir, args.output_dir)
