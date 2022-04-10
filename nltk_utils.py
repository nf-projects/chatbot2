import nltk
import numpy as np
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
    # This function takes a tokenized sentence, stems it,
    # then creates a list of 0s and 1s, where 1s are matches
    # to the "all_words" list.

    """
    Visualization:

    sentence = ["hello", "how", "are", "you]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag =   [  0,     1,     0,    1,     0,      0,       0, ]
    """

    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    #numpy zeros creates a matrix of zeros with the same shape as the list of words
    bag = np.zeros(len(all_words), dtype=np.float32)

    for index, word, in enumerate(all_words):
        if word in tokenized_sentence:
            # replace the 0 with a 1 if it's a match
            bag[index] = 1
    
    return bag
