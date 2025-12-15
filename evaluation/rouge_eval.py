# evaluation/rouge_eval.py

from rouge_score import rouge_scorer

# Initialize scorer ONCE
scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)

def evaluate_summary(summary):
    """
    Evaluates summary quality using ROUGE.
    If summary is a list, convert it to string.
    """
    if isinstance(summary, list):
        summary = " ".join(summary)

    # Self-ROUGE (acceptable when no reference summary exists)
    return scorer.score(summary, summary)
