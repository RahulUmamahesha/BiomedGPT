# CXR Description Pipeline (BiomedGPT)

Generates free-text descriptions of chest X-ray (CXR) images using BiomedGPT, a
vision-language transformer (OFA architecture). Built as a standalone modality
pipeline, intended to plug into a larger multimodal system later (in the style
of the MANGO paper's per-modality pipelines).

## Quick start

```bash
source venv/bin/activate
python -m pipeline.run_pipeline \
  --manifest data/iu_xray_sample/manifest.csv \
  --checkpoint checkpoints/iu-xray-finetuned \
  --output outputs/generated_descriptions.jsonl
```

Add `--sampling` to use nucleus sampling (top_p=0.9, temp=0.7) instead of beam
search. `--manifest` expects a CSV with `image_id,image_path,ground_truth_report`
columns (`ground_truth_report` is optional, used only for side-by-side printing).

## Architecture

```
CXR image ──▶ preprocessing ──▶ BiomedGPT (OFA) ──▶ vocab-restricted   ──▶ generated_descriptions.jsonl
             (resize 256²,      encoder-decoder      beam/sample decode      {image_id, description,
              normalize,        + ResNet101 visual                           ground_truth, checkpoint,
              BPE tokenize      backbone                                     timestamp}
              prompt)
```

`pipeline/infer.py` — `BiomedGPTCaptioner`: loads a checkpoint once, exposes
`describe(image_path) -> str`.
`pipeline/run_pipeline.py` — CLI: reads a manifest CSV, runs inference per row,
writes JSONL results.
`pipeline/config.py` — checkpoint path, prompt, image transform params,
generation params (beam and sampling presets).
`pipeline/convert_fairseq_checkpoint.py` — converts a fairseq-format BiomedGPT
`.pt` checkpoint into this project's HF-transformers checkpoint layout (see
below — needed because BiomedGPT's own fine-tuned checkpoints ship in fairseq
format only).

## Environment

- Python 3.10 venv (`venv/`) — the system Python (3.13) is too new for the
  OFA/fairseq stack. `python@3.10` came from Homebrew.
- `transformers` is **not** stock PyPI transformers — it's an editable install
  of `OFA-Sys/OFA`'s `feature/add_transformers` branch (`OFA/transformers/`),
  which adds the `OFAModel`/`OFATokenizer` classes BiomedGPT's checkpoints use.
  This avoids BiomedGPT's official fairseq + Python 3.7 install path entirely,
  which has C-extension build problems on Apple Silicon.
- CPU-only inference (no CUDA on this machine; MPS is available but untested).

### Two bugs found and fixed in the vendored OFA code

1. **`decoder.image_position_idx` size mismatch** (`OFA/transformers/src/transformers/models/ofa/modeling_ofa.py`,
   ~line 1365): this fork hardcodes a 768-element fill for a position-index
   buffer; BiomedGPT-base checkpoints need 769. One-line patch, documented
   inline. Without it, `OFAModel.from_pretrained` fails to load any
   BiomedGPT-base checkpoint.
2. **Stray `<code_NNNN>` tokens corrupting every caption** (`pipeline/infer.py`,
   `RestrictToTextVocab`): OFA's vocabulary packs image-codebook tokens
   (indices 50265–58456) and bbox-bin tokens (58457–59456) above the real text
   vocabulary (0–50264). Fairseq's own caption inference masks these out via a
   `constraint_range` argument; this HF port's plain `generate()` doesn't, so
   without an explicit `LogitsProcessor` banning that range, the model
   sometimes emits a codebook token as the first word of a caption, corrupting
   it. Fixed by adding `RestrictToTextVocab`, applied via `logits_processor=`
   in every `generate()` call.

## Checkpoints

| Directory | Source | Format | Notes |
|---|---|---|---|
| `checkpoints/instruct-biomedgpt-base` | `PanaceaAI/instruct-biomedgpt-base` (HF Hub) | HF-native | Generic instruction-tuned BiomedGPT-base (182M params), not CXR-specific |
| `checkpoints/fairseq_iu_xray/iu_xray.pt` | BiomedGPT repo's `checkpoints.md` (Dropbox) | fairseq | Raw fine-tuned checkpoint, fine-tuned on IU X-ray specifically. Not directly loadable by our `transformers` fork |
| `checkpoints/iu-xray-finetuned` | Converted from the above via `pipeline/convert_fairseq_checkpoint.py` | HF-native | Same weights as `iu_xray.pt`, in this project's loadable format |

`convert_fairseq_checkpoint.py` renames ~128 state-dict keys (fairseq's
`encoder_attn`/`attn_ln`/`ffn_layernorm` naming → HF's `cross_attn`/
`self_attn_mid_layer_norm`/`ffn_layer_norm`) and validates the result loads
with `strict=True` before writing it out. See the module docstring for the
full rename table. It's checkpoint-agnostic for any BiomedGPT-base-sized
fairseq checkpoint (e.g. the SLAKE/PathVQA/ROCO/Peir-Gross fine-tunes listed
in BiomedGPT's `checkpoints.md`) — only tested against `iu_xray.pt` so far.

## Test data

- `data/iu_xray_sample/` — 15 images, initial qualitative-review sample.
- `data/iu_xray_sample_100/` — 100 images, used for the quantitative pass below.

Both streamed from `X-iZhang/IU-Xray-RRG` on Hugging Face (IU X-ray/OpenI
dataset, no credentialing required, unlike MIMIC-CXR). `manifest.csv` has
`image_id,image_path,ground_truth_report`.

## Evaluation

`pipeline/evaluate.py` scores a `run_pipeline.py` output on: BLEU, ROUGE-L,
exact-duplicate rate (the mode-collapse symptom), and a negation-aware
clinical-keyword hallucination/miss rate (checks a small preceding-word window
for cues like "no"/"without" before each keyword hit, since radiology text is
dominated by negated findings — a heuristic, not full clinical NLP negation
detection).

```bash
python -m pipeline.evaluate outputs/generated_descriptions.jsonl
```

## Quality findings

Neither checkpoint is clinically reliable as-is. This is a real limitation of
BiomedGPT-base on radiology report generation, not a pipeline bug — mechanics
are validated end-to-end (correct preprocessing, correct prompt, vocab-masked
decoding, verified checkpoint conversion, `evaluate.py` cross-checked against
manual reading of individual examples).

**n=100** (`outputs/n100_*.jsonl`):

| Checkpoint | Decoding | Duplicate rate | Hallucination rate | Miss rate | BLEU | ROUGE-L |
|---|---|---|---|---|---|---|
| `instruct-biomedgpt-base` | beam (5) | 2% | 1.5% | 93.2% | 0.56 | 0.126 |
| `iu-xray-finetuned` | beam (5) | **67%** | 8.4% | 75.0% | 6.90 | 0.256 |
| `iu-xray-finetuned` | sampling (top_p=0.9, temp=0.7) | 48% | 6.4% | 79.5% | 6.63 | 0.258 |

(n=15 pilot showed the same pattern at smaller scale: 0%/60%/47% duplicate
rates respectively — going to n=100 made the fine-tuned checkpoint's collapse
*worse*, not better, ruling out small-sample noise as the explanation.)

**Reading this table:** base-instruct's near-zero hallucination rate isn't
really a strength — it almost never commits to a specific finding at all (93%
miss rate on real findings), so it "wins" on hallucination mainly by staying
generic. The fine-tuned checkpoint says more specific, real-sounding things
(much better BLEU/ROUGE) but is wrong more often and mode-collapses onto ~15
canned templates two-thirds of the time. Sampling reliably trades some of that
duplication and hallucination away, at a small miss-rate cost.

**Takeaway (revised after the fine-tuning experiment below):** the fine-tuned
checkpoint's mode collapse is a training-data/training-objective issue (IU
X-ray's fine-tuning set is dominated by normal cases; MLE training rewards
always predicting the majority template) — sampling only partially masks it.
A training-time fix (rebalanced data) does fix the collapse itself, but does
**not** straightforwardly translate into better clinical accuracy — see below.

### GPU replication (n=50, A100)

`notebooks/biomedgpt_cxr_gpu_eval.ipynb` reran `iu-xray-finetuned` on a
completely different environment (Colab A100 vs. local CPU): **60% exact-duplicate
rate**, matching the local n=15 (60%) and n=100 (67%) results closely. Three
sample sizes, two independent environments, same collapse rate — this ruled
out small-sample noise or a local-environment artifact as the explanation.
The notebook also runs all three original configurations on the **full
~3,307-image** IU-Xray-RRG test split, plus a richer negation-aware
14-category clinical-finding F1 approximating CheXbert's label schema (a
regex heuristic over the standard category names, *not* the real trained
CheXbert BERT labeler). Set `MAX_IMAGES` in the notebook for a quick smoke
test before the full run.

## Fine-tuning experiment: does fixing mode collapse fix accuracy?

`notebooks/biomedgpt_finetune_colab.ipynb` re-does the IU-X-ray fine-tune from
`instruct-biomedgpt-base` (not continuing from the already-collapsed
checkpoint), using a `WeightedRandomSampler` that oversamples training reports
containing an actual clinical finding (detected via the same negation-aware
keyword logic as `evaluate.py`) — directly targeting the class imbalance
behind the collapse. Trained 6 epochs on `dz-osamu/IU-Xray`'s real train
split (2,069 examples; distinct from `X-iZhang/IU-Xray-RRG`, which is
test-only), on a Colab A100, val loss converging cleanly (2.87→2.47, correctly
not saving the epoch-6 checkpoint once val loss ticked back up).

Result, n=100 on held-out validation images, all beam search:

| Checkpoint | BLEU | ROUGE-L | Duplicate rate | Unique outputs | CheXpert-style micro-F1 |
|---|---|---|---|---|---|
| `instruct-biomedgpt-base` | 0.56 | 0.126 | 2% | 98 | ~low (93% miss rate on findings) |
| `iu-xray-finetuned` | 6.90 | 0.256 | 67% | 15 | **0.243** |
| `iu-xray-finetuned` (sampling) | 6.63 | 0.258 | 48% | 33 | **0.265** |
| `iu-xray-rebalanced` (this experiment) | **12.85** | **0.300** | **22%** | 23 | **0.192** |

The rebalanced checkpoint **fixed the diversity problem it targeted** (67%→22%
duplication) and produced text that reads more like a real report (BLEU
more than doubled). But **clinical-finding accuracy went down, not up**
(micro-F1 0.243→0.192) — per-category breakdown shows `Lung Opacity` detection
dropped to 0.0 (from 0.25-0.4) and `No Finding` dropped to 0.352 (from
0.45-0.5); only `Pneumothorax` improved (0→0.057).

**Conclusion: mode collapse and clinical hallucination are separable failure
modes, not two symptoms of the same underlying problem.** Rebalancing the
training data to punish the "always predict normal" shortcut successfully
stopped the model from taking that shortcut — but the model filled the gap
with more confident, specific-sounding claims that are often still wrong,
rather than becoming more accurate. Better BLEU/ROUGE despite worse
clinical-finding F1 is the concrete version of the general warning from
earlier in this document: text-overlap metrics can move in the opposite
direction from actual correctness on this task, so both need to be reported
together, not just one. This is a legitimate, useful research finding for a
182M-parameter model on this dataset, even though it did not produce a
strictly-better checkpoint than what already existed.

## Known limitations / not yet built

- The `iu-xray-rebalanced` checkpoint from the fine-tuning experiment lives
  only in the Colab session that trained it (downloaded as a zip) — it hasn't
  been pulled into this project's local `checkpoints/` folder, since the
  experiment's conclusion was "not a strict improvement," not "adopt this."
- No retry/rate-limiting/checkpointing for large batch runs (current
  `run_pipeline.py` is a simple single-process loop with a try/except per
  image).
- CPU-only locally; GPU path only exists in the Colab notebooks so far.
- The clinical-keyword metric (both local `evaluate.py` and the notebooks'
  CheXpert-style version) is a regex + negation-window heuristic, not a
  validated clinical NLP tool — treat it as directionally informative, not
  a certified score.
- Only one rebalancing strategy was tried (oversampling via
  `WeightedRandomSampler`, 6 epochs, no hyperparameter search). Untried:
  label smoothing changes, lower learning rate, more epochs with early
  stopping, or combining rebalancing with sampling-based decoding.
- `checkpoints.md` in BiomedGPT's own repo lists SLAKE, PathVQA, VQA-RAD,
  Peir-Gross, and ROCO fine-tuned checkpoints too — only IU X-ray's was
  converted and tested here.
