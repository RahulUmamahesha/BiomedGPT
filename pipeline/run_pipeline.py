import argparse
import csv
import datetime
import json
import os

import torch

from .infer import BiomedGPTCaptioner
from . import config


def run(manifest_path, output_path, checkpoint_dir=config.CHECKPOINT_DIR, sampling=False, seed=42):
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    generation_params = config.SAMPLING_GENERATION_PARAMS if sampling else config.GENERATION_PARAMS

    print(f"Loaded {len(rows)} rows from {manifest_path}")
    print(f"Loading BiomedGPT model from {checkpoint_dir}...")
    print(f"Generation params: {generation_params}")
    captioner = BiomedGPTCaptioner(checkpoint_dir=checkpoint_dir, generation_params=generation_params)
    print("Model loaded. Running inference...")

    if sampling:
        torch.manual_seed(seed)  # seeded once up front, not per-image, so each image draws fresh samples

    results = []
    for i, row in enumerate(rows):
        image_path = row["image_path"]
        try:
            description = captioner.describe(image_path)
        except Exception as e:
            print(f"[{i+1}/{len(rows)}] FAILED on {row['image_id']}: {e}")
            continue

        result = {
            "image_id": row["image_id"],
            "image_path": image_path,
            "generated_description": description,
            "ground_truth_report": row.get("ground_truth_report", ""),
            "checkpoint": os.path.basename(checkpoint_dir),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        results.append(result)
        print(f"[{i+1}/{len(rows)}] {row['image_id']}")
        print(f"  generated: {description}")
        print(f"  truth:     {row.get('ground_truth_report', '')}")

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(results)} results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=os.path.join(config.PROJECT_ROOT, "data", "iu_xray_sample", "manifest.csv"))
    parser.add_argument("--output", default=os.path.join(config.PROJECT_ROOT, "outputs", "generated_descriptions.jsonl"))
    parser.add_argument("--checkpoint", default=config.CHECKPOINT_DIR)
    parser.add_argument("--sampling", action="store_true", help="use nucleus sampling instead of beam search")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.manifest, args.output, checkpoint_dir=args.checkpoint, sampling=args.sampling, seed=args.seed)
