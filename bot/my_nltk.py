import nltk
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import numpy as np

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return ps.stem(word.lower())

def bag_of_words(tokenised_sentence, all_words):
    tokenised_sentence = [stem(w) for w in tokenised_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenised_sentence:bag[idx] = 1.0

    return bag


         
