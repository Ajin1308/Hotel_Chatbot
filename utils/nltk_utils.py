import nltk
import numpy as np
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    '''
    we are tokenizing(sentence into words) using the inbuilt function word_tokenize

    '''
    return nltk.word_tokenize(sentence)

def stem(word):
    '''
    we are getting the root form of a word using this function using the inbuilt PorterStemmer() module.
    stemming helps in reducing dimensionality of the data

    '''
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    '''
    This fuction creates a bag of words representaion.
    Representing each word in the vocabulary as 0->absent or 1->present (binary).
    Returns the bag if a word from all_words present in the tokenized sentence
    '''
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

