import nltk
# If it gives errors:
# nltk.download('punkt')

# for stemming
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    pass

string = "How long does shipping take?"
print(string)
print(tokenize(string))

words = ["Organize", "shipping", "take"]
stemmed_words = [stem(word) for word in words]
print(stemmed_words)
