from sklearn.feature_extraction.text import TfidfVectorizer
import heapq
import re
import nltk
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.sentiment import SentimentIntensityAnalyzer

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('words')
# nltk.download('maxent_ne_chunker')
# nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

# Sentiment Analysis
def sent_analysis(text):

    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)['compound']

    if score >= 0.05:
        sentiment = "Positive Sentiment"
    elif score <= -0.05:
        sentiment = "Negative Sentiment"
    else:
        sentiment = "Neutral Sentiment"

    return score, sentiment


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    if treebank_tag.startswith('N'):
        return wordnet.NOUN
    if treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    # Step 0: Remove HTML tags
    soup = BeautifulSoup(text, 'html.parser')
    text_without_html = soup.get_text()

    # Filter common sentences
    location_pattern = re.compile(r'^[A-Z\s\.,]+? \(\w+\) - ')
    copyright_pattern = re.compile(r'^Copyright \d{4} [^\n]+')
    photo_caption_pattern = re.compile(r'\([^\n]+\)(\.|,)')

    # Remove location indicators
    text_without_html = re.sub(location_pattern, '', text_without_html)

    # Remove copyright information
    text_without_html = re.sub(copyright_pattern, '', text_without_html)

    # Remove photo captions
    text_without_html = re.sub(photo_caption_pattern, '', text_without_html)
    
    # Step 1: Expand contractions
    contractions = {
        "n't" : "not",
        "'s" : "is",
        "'re" : "are",
        "'m" : "am",
        "'ll" : "will",
        "'d" : "would",
        "'ve" : "have"
    }
    expanded_text = ' '.join([contractions.get(word, word) for word in word_tokenize(text_without_html)])

    # Step 2: Tokenize sentences
    sentences = sent_tokenize(expanded_text)

    # Step 3: Tokenize words in sentences
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

    # Step 4: Lowercasing
    lowercased_tokens = [
        [word.lower() for word in tokens]
        for tokens in tokenized_sentences
    ]
    
    # Step 5: Remove Punctuation
    filtered_tokens = [
        [word for word in tokens if word.isalpha()]
        for tokens in lowercased_tokens
    ]

    # Step 6: Stopwords Removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [
        [word for word in tokens if word not in stop_words]
        for tokens in filtered_tokens
    ]

    # Step 7: Part of Speech Tagging (POS)
    pos_tags = [pos_tag(tokens) for tokens in filtered_tokens]

    # Step 8: Named Entity Recognition (NER)
    def extract_entities(pos_tags):
        entities = []
        for sent_tags in pos_tags:
            chunked = ne_chunk(sent_tags)
            for subtree in chunked:
                if isinstance(subtree, Tree) and subtree.label() == 'NE':
                    entity = " ".join([token[0] for token in subtree.leaves()])
                    entities.append(entity)
        return entities

    # Extract named entities
    entities = extract_entities(pos_tags)

    # Step 9: Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemma_tokens = [
        [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tag_i]
        for pos_tag_i in pos_tags
    ]

    return sentences, filtered_tokens, pos_tags, entities, lemma_tokens


def generate_summary(text, num_sentences=5):
    # Call preprocess_text function to obtain processed components
    sentences, filtered_tokens, pos_tags, entities, lemma_tokens = preprocess_text(text)

    # TD-IDF Vectorization - Create vectorizer and apply it to the lemmatized tokens to transform it into numerical vectors
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    vectors = vectorizer.fit_transform([' '.join(tokens) for tokens in lemma_tokens])

    # Calculate sentence similarity using cosine similarity
    sentence_similarity_matrix = cosine_similarity(vectors, vectors)

    # Sentence scores calculation - Sum the cosine similarity scores along each row to get the scores for each sentence. Use heapq.nlargest to get the indices of the top sentences
    sentence_scores = sentence_similarity_matrix.sum(axis=1)
    ranked_sentences = heapq.nlargest(num_sentences, range(len(sentence_scores)), sentence_scores.__getitem__)

    # Extract summary sentences
    summary = [sentences[i] for i in sorted(ranked_sentences)]

    return " ".join(summary)

def generate_abstractive_summary(text, max_length=300, min_length=100, length_penalty=2.0):
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Tokenize and generate summary
    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(input_ids, max_length=max_length, min_length=min_length, length_penalty=length_penalty, num_beams=4, early_stopping=True)

    # Decode summary
    abs_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return abs_summary


# if __name__=="__main__":
#     text = '''
# '''
#     summary = generate_summary(text)
#     print(f"Text lenth: {len(text)}")
#     print(f"Summary lenth: {len(summary)}")
#     print(summary)