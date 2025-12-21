from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)

def evaluate_summary(generated_summary, reference_summary):
    """
    Compare model-generated summary with gold reference summary
    """

    # Safety: handle list inputs
    if isinstance(generated_summary, list):
        generated_summary = " ".join(generated_summary)

    if isinstance(reference_summary, list):
        reference_summary = " ".join(reference_summary)

    scores = scorer.score(
        reference_summary,     # 📌 reference FIRST
        generated_summary      # 📌 candidate SECOND
    )

    # Optional: convert to readable dict
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    }
