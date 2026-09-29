import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "instruct-biomedgpt-base")

# Exact prompt BiomedGPT's own fairseq caption task uses (data/mm_data/caption_dataset.py)
CAPTION_PROMPT = " what does the image describe?"

IMAGE_RESOLUTION = 256
IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD = [0.5, 0.5, 0.5]

DEVICE = "cpu"  # matches the CPU-only target environment; untested on MPS/CUDA

# Where real BPE text ends in OFA's vocab; above this are image-codebook/bbox tokens
# that generate() can otherwise emit mid-caption (fairseq masks these, this HF port doesn't).
TEXT_VOCAB_SIZE = 50265

GENERATION_PARAMS = {
    "num_beams": 5,
    "no_repeat_ngram_size": 3,
    "max_length": 64,
}

# Beam search always returns the single highest-probability sequence, which is what let
# iu-xray-finetuned collapse onto a handful of majority-template outputs; sampling explores
# lower-probability continuations instead.
SAMPLING_GENERATION_PARAMS = {
    "do_sample": True,
    "num_beams": 1,
    "top_p": 0.9,
    "temperature": 0.7,
    "no_repeat_ngram_size": 3,
    "max_length": 64,
}
