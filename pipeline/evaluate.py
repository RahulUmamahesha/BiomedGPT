"""
Scores a run_pipeline.py output (JSONL of {generated_description, ground_truth_report, ...})
against three angles:
  - BLEU / ROUGE-L: standard text-overlap metrics.
  - exact-duplicate rate: how often the exact same string gets generated for different images
    (the mode-collapse symptom found in manual review).
  - clinical keyword agreement: for a fixed list of CXR finding terms, whether the generated
    text and the ground truth AGREE on whether that finding is present, absent (negated), or
    unmentioned. This is negation-aware (checks a few words before each keyword hit for cues
    like "no"/"without") because radiology text is dominated by negated findings ("no
    pneumothorax") -- naive substring presence would massively overcount hallucinations.
    This is a simple heuristic (a small preceding-word window), not full clinical NLP negation
    detection (e.g. NegEx) -- it will misjudge negation scope in more complex sentences.
"""
import argparse
import json
import re
from collections import Counter

import sacrebleu
from rouge_score import rouge_scorer

KEYWORDS = {
    "pneumothorax": ["pneumothorax"],
    "effusion": ["effusion"],
    "consolidation": ["consolidation"],
    "atelectasis": ["atelectasis"],
    "cardiomegaly": ["cardiomegaly", "heart is enlarged", "cardiac silhouette is enlarged", "heart size is enlarged"],
    "edema": ["edema"],
    "opacity": ["opacity", "opacities", "opacification"],
    "pneumonia": ["pneumonia"],
    "emphysema": ["emphysema"],
    "nodule_or_mass": ["nodule", "mass"],
    "fracture": ["fracture"],
}

NEGATION_CUES = ["no ", "without ", "not ", "free of ", "negative for ", "absence of ", "resolution of ", "resolved "]
NEGATION_WINDOW_CHARS = 40  # how far back to look for a negation cue before a keyword hit


def find_assertions(text):
    """Returns {keyword: 'positive'|'negative'|'absent'} for the fixed KEYWORDS list."""
    text = text.lower()
    result = {}
    for keyword, variants in KEYWORDS.items():
        status = "absent"
        for variant in variants:
            for m in re.finditer(re.escape(variant), text):
                window_start = max(0, m.start() - NEGATION_WINDOW_CHARS)
                window = text[window_start:m.start()]
                if any(cue in window for cue in NEGATION_CUES):
                    status = "negative"
                else:
                    status = "positive"
                    break  # a positive mention anywhere wins over an earlier negative one
            if status == "positive":
                break
        result[keyword] = status
    return result


def keyword_agreement(generated_list, truth_list):
    """Aggregate TP/FP/FN/TN across all keywords x all examples.
    FP = generated asserts positive, truth doesn't (absent or negative) -> hallucination.
    FN = truth asserts positive, generated doesn't -> missed finding.
    """
    counts = Counter()
    for gen_text, truth_text in zip(generated_list, truth_list):
        gen_assertions = find_assertions(gen_text)
        truth_assertions = find_assertions(truth_text)
        for keyword in KEYWORDS:
            gen_pos = gen_assertions[keyword] == "positive"
            truth_pos = truth_assertions[keyword] == "positive"
            if gen_pos and truth_pos:
                counts["TP"] += 1
            elif gen_pos and not truth_pos:
                counts["FP"] += 1
            elif not gen_pos and truth_pos:
                counts["FN"] += 1
            else:
                counts["TN"] += 1
    return counts


def evaluate(results_path):
    results = [json.loads(l) for l in open(results_path)]
    generated = [r["generated_description"] for r in results]
    truth = [r["ground_truth_report"] for r in results]

    bleu = sacrebleu.corpus_bleu(generated, [truth])
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_scores = [scorer.score(t, g)["rougeL"].fmeasure for g, t in zip(generated, truth)]
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

    dup_counts = Counter(generated)
    most_common_text, most_common_count = dup_counts.most_common(1)[0]
    duplicate_rate = most_common_count / len(generated)

    kw_counts = keyword_agreement(generated, truth)
    hallucination_rate = kw_counts["FP"] / (kw_counts["FP"] + kw_counts["TN"]) if (kw_counts["FP"] + kw_counts["TN"]) else 0.0
    miss_rate = kw_counts["FN"] / (kw_counts["FN"] + kw_counts["TP"]) if (kw_counts["FN"] + kw_counts["TP"]) else 0.0

    summary = {
        "n": len(results),
        "bleu": round(bleu.score, 2),
        "rouge_l_f1": round(avg_rouge_l, 4),
        "exact_duplicate_rate": round(duplicate_rate, 3),
        "unique_outputs": len(dup_counts),
        "keyword_true_positive": kw_counts["TP"],
        "keyword_false_positive_hallucination": kw_counts["FP"],
        "keyword_false_negative_missed": kw_counts["FN"],
        "keyword_true_negative": kw_counts["TN"],
        "keyword_hallucination_rate": round(hallucination_rate, 3),
        "keyword_miss_rate": round(miss_rate, 3),
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="path to a run_pipeline.py JSONL output file")
    args = parser.parse_args()
    summary = evaluate(args.results)
    print(json.dumps(summary, indent=2))
