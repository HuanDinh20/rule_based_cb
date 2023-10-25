import re
import random
import string
import bs4 as bs

from urllib import request
import nltk

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

get_link = request.urlopen('https://vi.wikipedia.org/wiki/Tr%C3%AD_tu%E1%BB%87_nh%C3%A2n_t%E1%BA%A1o')
get_link = get_link.read()
data = bs.BeautifulSoup(get_link, "lxml")

paragraph = data.find_all('p')
data_text = ''

for para in paragraph:
    data_text += para.text

data_text = re.sub(r'\[[0-9]*\]', ' ', data_text)
data_text = re.sub(r'\s+', ' ', data_text)

data_sentences = nltk.sent_tokenize(data_text)
data_words = nltk.word_tokenize(data_text)

wnlemmatizer = nltk.stem.WordNetLemmatizer()


def perform_lemmatization(tokens):
    return [wnlemmatizer.lemmatize(token) for token in tokens]


def get_processed_text(document):
    return perform_lemmatization(nltk.word_tokenize(document.lower().translate(punctuation_removal)))


def generate_greeting_response(greeting):
    for token in greeting.split():
        if token.lower() in greeting_inputs:
            return random.choice(greeting_responses)

def generate_response(user_input):
    bot_response = ''
    data_sentences.append(user_input)

    word_vectorizer = TfidfVectorizer(tokenizer=get_processed_text)
    all_word_vectors = word_vectorizer.fit_transform(data_sentences)
    similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors)
    similar_sentence_number = similar_vector_values.argsort()[0][-2]

    matched_vector = similar_vector_values.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-2]

    if vector_matched == 0:
        bot_response = bot_response + "I am sorry, I could not understand you"
        return bot_response
    else:
        bot_response = bot_response + data_sentences[similar_sentence_number]
        return bot_response

punctuation_removal = dict((ord(punctuation), None) for punctuation in string.punctuation)

greeting_inputs = ("hey", "good morning", "good evening", "morning", "evening", "hi", "whatsup")
greeting_responses = ["hey", "hey hows you?", "*nods*", "hello, how you doing", "hello", "Welcome, I am good and you"]


continue_dialogue = True
print("Hello, I am from AI Science. You can ask me any question regarding AI:")
while(continue_dialogue == True):
    human_text = input()
    human_text = human_text.lower()
    if human_text != 'bye':
        if human_text == 'thanks' or human_text == 'thank you very much' or human_text == 'thank you':
            continue_dialogue = False
            print("AI Sciences: Most welcome")
        else:
            if generate_greeting_response(human_text) != None:
                print("AI Sciences: " + generate_greeting_response(human_text))
            else:
                print("AI Sciences: ", end="")
                print(generate_response(human_text))
                data_sentences.remove(human_text)
    else:
        continue_dialogue = False
        print("AI Sciences: Good bye and take care of yourself...")

