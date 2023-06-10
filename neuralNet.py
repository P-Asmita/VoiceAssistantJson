import numpy as np
import nltk #natural language toolkit
from nltk.stem.porter import PorterStemmer

#creating neurons i.e. get input-> processing -> forwarding data

Stemmer = PorterStemmer()

#tokenize
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

#convert to words
def stem(word):
    return Stemmer.stem(word.lower())

#packed in a bag of words to be trained by our neural network later
def bag_of_words(tokenized_sentence,words):
    sentence_word = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words),dtype=np.float32)

    for index, w in enumerate(words):
        if w in sentence_word:
            bag[index] = 1
        
    return bag


