# CXR Description Pipeline (BiomedGPT)

Generates free-text descriptions of chest X-ray images using BiomedGPT (a
vision-language transformer). Intended as one modality pipeline in a larger
multimodal system.

## Quick start

```bash
pip install -r requirements.txt
# then set up the OFA-Sys transformers fork and apply the two patches below
python -m pipeline.run_pipeline \
  --manifest data/iu_xray_sample/manifest.csv \
  --checkpoint checkpoints/iu-xray-finetuned \
  --output outputs/generated_descriptions.jsonl
```

Add `--sampling` for nucleus sampling instead of beam search.

## Architecture

```
CXR image → preprocess (resize/normalize) → BiomedGPT (OFA) → vocab-restricted decode → description
```

- `pipeline/infer.py` — `BiomedGPTCaptioner`: loads a checkpoint, exposes `describe(image_path)`
- `pipeline/run_pipeline.py` — CLI: batch-runs a manifest CSV to JSONL
- `pipeline/convert_fairseq_checkpoint.py` — converts BiomedGPT's fairseq checkpoints to this project's HF format
- `pipeline/evaluate.py` — scores a run's output (BLEU, ROUGE-L, duplicate rate, negation-aware clinical-keyword F1)

## Setup

Requires Python 3.10 (not stock 3.13/3.7). `transformers` is not the PyPI
package — it's `OFA-Sys/OFA`'s `feature/add_transformers` fork, which adds the
`OFAModel`/`OFATokenizer` classes BiomedGPT needs (this avoids BiomedGPT's
official fairseq install path, which has build problems on Apple Silicon).

```bash
git clone --single-branch --branch feature/add_transformers https://github.com/OFA-Sys/OFA.git
pip install -e ./OFA/transformers/
```

Two bugs in that fork need patching before use (both documented inline where
patched, in `pipeline/infer.py` and via a one-line edit to `modeling_ofa.py`):
1. A hardcoded buffer size (768) that's one short of what BiomedGPT-base
   checkpoints need (769) — fails checkpoint loading otherwise.
2. Missing vocabulary restriction during generation — without it, the model
   sometimes emits stray image-codebook tokens mid-caption.

## Checkpoints

| Directory | Source |
|---|---|
| `checkpoints/instruct-biomedgpt-base` | `PanaceaAI/instruct-biomedgpt-base` (HF Hub), generic, not CXR-specific |
| `checkpoints/iu-xray-finetuned` | BiomedGPT's own IU-X-ray fine-tune, converted from fairseq via `convert_fairseq_checkpoint.py` |

## Findings

Neither checkpoint is clinically reliable as-is — this is a real limitation of
BiomedGPT-base on this task, not a pipeline bug (mechanics are validated
end-to-end: correct preprocessing, verified checkpoint conversion, evaluation
metrics cross-checked against manual reading).

| Checkpoint | BLEU | ROUGE-L | Duplicate rate | Clinical-finding F1 |
|---|---|---|---|---|
| `instruct-biomedgpt-base` | 0.56 | 0.126 | 2% | low (93% miss rate on real findings) |
| `iu-xray-finetuned` (beam) | 6.90 | 0.256 | **67%** | 0.243 |
| `iu-xray-finetuned` (sampling) | 6.63 | 0.258 | 48% | 0.265 |

- The generic checkpoint rarely commits to specific findings — safe, but not useful.
- The fine-tuned checkpoint mode-collapses onto a handful of canned "normal"
  templates for 60-67% of images regardless of content — confirmed across
  n=15/50/100 samples on two independent environments (local CPU, Colab A100),
  ruling out small-sample noise.

**Fine-tuning experiment** (`notebooks/biomedgpt_finetune_colab.ipynb`):
re-trained from the generic checkpoint using a rebalanced sampler that
oversamples reports with real findings, to directly target the collapse.

| | BLEU | ROUGE-L | Duplicate rate | Clinical-finding F1 |
|---|---|---|---|---|
| `iu-xray-rebalanced` | **12.85** | **0.300** | **22%** | 0.192 |

This fixed the collapse (67%→22%) and improved text fluency, but clinical
accuracy went *down* (F1 0.243→0.192) — the model stopped defaulting to safe
generic templates but filled the gap with more confident, still-often-wrong
specific claims. **Conclusion: mode collapse and clinical hallucination are
separable failure modes** — fixing one doesn't fix the other, and text-overlap
metrics (BLEU/ROUGE) can improve while accuracy gets worse.

## Notebooks

- `notebooks/biomedgpt_cxr_gpu_eval.ipynb` — GPU replication of the findings
  above at larger scale (full ~3,307-image test set), self-contained.
- `notebooks/biomedgpt_finetune_colab.ipynb` — the rebalancing experiment,
  self-contained, includes training + evaluation.

## Known limitations

- `iu-xray-rebalanced` isn't adopted as the default checkpoint (not a strict
  improvement) and isn't included in `checkpoints/` — it only exists in the
  Colab session that trained it.
- Clinical-keyword scoring is a negation-aware regex heuristic, not a
  validated clinical NLP tool (e.g. not the real CheXbert labeler).
- No retry/checkpointing for large batch runs.
- Only one rebalancing strategy tried (oversampling, 6 epochs, no
  hyperparameter search).
