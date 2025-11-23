import re
import textstat

TRANSITIONS = [
    "however","therefore","moreover","in contrast","for example",
    "because","consequently","on the other hand","furthermore"
]

def basic_features(text, prompt=None):
    text_clean = re.sub(r"\s+", " ", text.strip())
    words = text_clean.split()
    num_words = len(words)
    num_sentences = max(1, text_clean.count(".") + text_clean.count("!") + text_clean.count("?"))

    readability = textstat.flesch_reading_ease(text_clean)
    reading_level = textstat.text_standard(text_clean, float_output=True)

    transition_count = sum(text_clean.lower().count(t) for t in TRANSITIONS)
    avg_sentence_len = num_words / num_sentences

    prompt_sim = 0.0
    if prompt and len(prompt.strip()) > 0:
        p_words = set(re.findall(r"[a-zA-Z']+", prompt.lower()))
        a_words = set(re.findall(r"[a-zA-Z']+", text_clean.lower()))
        prompt_sim = len(p_words & a_words) / max(1, len(p_words))

    return {
        "num_words": num_words,
        "num_sentences": num_sentences,
        "avg_sentence_len": avg_sentence_len,
        "readability": readability,
        "reading_level": reading_level,
        "transition_count": transition_count,
        "prompt_similarity": prompt_sim
    }