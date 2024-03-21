import spacy
from collections import defaultdict

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = defaultdict(int)
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])

    def addSentences(self, sentences):
        for doc in self.nlp.pipe(sentences, batch_size=50):
            for token in doc:
                self.addWord(token.text)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
        self.word2count[word] += 1

if __name__ == "__main__":
    # Create a new Lang object for English
    english = Lang('English')

    # Add some sentences to the Lang object
    sentences = [
        "Hello, how are you?",
        "I'm fine, thank you.",
        "And you?",
        "I'm also fine."
    ]
    english.addSentences(sentences)

    # Print the word2index, word2count, and index2word dictionaries
    print("word2index:", english.word2index)
    print("word2count:", english.word2count)
    print("index2word:", english.index2word)