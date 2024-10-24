import json
import numpy as np
import torch
import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
chat_history_file = r"artifacts/chat_history.json"


# Utility functions

def bag_of_words(tokenized_sentence, all_words):
    """
    Create a bag of words representation (binary).
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


def tokenize(sentence):
    """
    Tokenize a sentence into words.
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    Stem a word to its root form.
    """
    return stemmer.stem(word.lower())


def save_chat_history(chat_history):
    """
    Save chat history to a file.
    """
    with open(chat_history_file, "w") as file:
        json.dump(chat_history, file)


def load_chat_history():
    """
    Load chat history from a file.
    """
    try:
        with open(chat_history_file, "r") as file:
            data = file.read()
            if data:
                return json.loads(data)
            else:
                return []  # Return an empty list if file is empty
    except (FileNotFoundError, json.JSONDecodeError):
        return []  # Return an empty list if file doesn't exist or contains invalid JSON


def preprocess_user_input(user_input, all_words, device):
    """
    Preprocess user input by tokenizing, applying bag of words, and converting to tensor.
    """
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    return X


def predict(output, tags):
    """
    Predict the tag with the highest probability from the model output.
    """
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    return tag, prob
