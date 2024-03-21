import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.translate.bleu_score import sentence_bleu
import sacrebleu
from sacrebleu import sentence_bleu, corpus_bleu


# nltk.download('punkt')
# nltk.download('vader_lexicon')


def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)['compound']

    if score >= 0.05:
        sentiment = "Positive Sentiment"
    elif score <= -0.05:
        sentiment = "Negative Sentiment"
    else:
        sentiment = "Neutral Sentiment"

    return score, sentiment


def compute_bleu(reference, candidate):
    """
    Compute BLEU score using sacrebleu library.

    Args:
    reference (str or list of str): Reference translation(s).
    candidate (str or list of str): Candidate translation(s).

    Returns:
    float: BLEU score.
    """
    # If inputs are provided as strings, convert them to lists
    if isinstance(reference, str):
        reference = [reference]
    if isinstance(candidate, str):
        candidate = [candidate]

    # Compute BLEU score
    bleu = sacrebleu.corpus_bleu(candidate, [reference])
    
    # Return BLEU score
    return bleu.score

# Example usage:
# reference = "The quick brown fox jumps over the lazy dog."
# candidate = "The fast brown fox jumps over the lazy dog."
# bleu_score = compute_bleu(reference, candidate)
# print("BLEU score:", bleu_score)
