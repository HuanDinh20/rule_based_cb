import string
import json
import random

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_text(document):
    punctuation_removal = dict((ord(punctuation), None) for punctuation in string.punctuation)
    punctuation_removal_doc = document.lower().translate(punctuation_removal)
    tokens = nltk.word_tokenize(punctuation_removal_doc)
    wordnet_lemma = nltk.stem.WordNetLemmatizer()
    return [wordnet_lemma.lemmatize(token) for token in tokens]


def greeting_response(greeting_responses):
    return random.choice(greeting_responses)


def generate_response(user_input, questions, question2answer):
    bot_response = ''
    data_sentences = []
    data_sentences.extend(questions)
    data_sentences.extend([user_input])

    word_vectorizer = TfidfVectorizer(tokenizer=preprocess_text)
    all_word_vectors = word_vectorizer.fit_transform(data_sentences)
    similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors)
    similar_sentence_number = similar_vector_values.argsort()[0][-2]

    matched_vector = similar_vector_values.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-2]

    if vector_matched == 0:
        bot_response = bot_response + "Xin lỗi tôi chưa hiểu câu hỏi của bạn"
        return bot_response
    else:
        similar_question = data_sentences[similar_sentence_number]
        bot_response = bot_response + question2answer[similar_question]
        return bot_response


with open("data/mobi_qa.json", "r", encoding='utf-8') as f:
    mobi_qa = json.load(f)

question_list = list(mobi_qa.keys())
question_text = ". ".join(question_list)

print(mobi_qa)

greeting_inputs = ("hey", "good morning", "good evening", "morning", "evening", "hi", "whatsup")
greeting_responses = ["hey", "hey hows you?", "*nods*", "hello, how you doing", "hello", "Welcome, I am good and you"]


continue_dialogue = True
print("Xin chào, tôi là trợ lý ảo, tôi sẽ trả lời câu hỏi của bạn:")

while(continue_dialogue == True):
    human_text = input()
    human_text = human_text.lower()
    if human_text != 'bye':
        if human_text == 'thanks' or human_text == 'thank you very much' or human_text == 'thank you':
            continue_dialogue = False
            print("AI Sciences: Most welcome")
        else:
            print("AI Sciences: " + generate_response(human_text, question_list, mobi_qa))
    else:
        continue_dialogue = False
        print("AI Sciences: Good bye and take care of yourself...")