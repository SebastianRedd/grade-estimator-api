def generate_feedback(feats, grade_level):
    tips = []

    if feats["num_words"] < 180 and grade_level in ["11","12","college"]:
        tips.append("Your response is short for this level. Add more evidence or analysis.")

    if feats["prompt_similarity"] < 0.20:
        tips.append("You may not be fully addressing the prompt. Re-check each part of the task.")

    if feats["transition_count"] < 2:
        tips.append("Add transition words (however, therefore, for example) to improve flow.")

    if feats["avg_sentence_len"] > 28:
        tips.append("Some sentences are long. Break them up for clarity.")

    if not tips:
        tips.append("Strong structure overall. Add one more concrete example to deepen your analysis.")

    return tips