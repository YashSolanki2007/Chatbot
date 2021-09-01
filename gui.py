# Imports
import json
import tensorflow as tf
import numpy as np
import math
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow import keras
import tkinter as tk
from tkinter import messagebox
import requests
import webbrowser
from datetime import datetime
import playsound
from gtts import gTTS


# Making the window
root = tk.Tk()
root.geometry("600x600")

# Making the response label
response_label = tk.Label(root, text='')
response_label.place(relx=0, rely=0.6, relwidth=1, relheight=0.15)

# Building the question label
question_label = tk.Label(root, text='')
question_label.place(relx=0, rely=0.4, relwidth=1, relheight=0.1)

# Important Variables
vocab_size = 800
embedding_dim = 17
max_length = 30
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
# Total size 32
training_size = 32
num_epochs = 175

# Defining the class names
class_names = ["Greeting", "Name", "Age",
               "Goodbye", "Profanity", "Search", "Time"]

# Loading in the json data
with open("conversation.json", 'r') as f:
    datastore = json.load(f)

# Making the seperate lists for the sentence and if it has profanity or not
sentences = []
labels = []

# Looping over and grabbing all elements listed above
for item in datastore:
    sentences.append(item['sentence'])
    labels.append(item['tag'])

# Creating the training and the testing data
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]
print(testing_labels)
print(training_labels)

print(training_sentences)

# Creating the keras tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

# Converting the texts to sequencies using the tokenizer
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(
    training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(
    testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

# Making the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(700, activation='relu'),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(124, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(7, activation='sigmoid')
])

# Loading the model
# model = keras.models.load_model("chatbot-model/")

# Compiling and fitting the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Fitting the model
history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels), verbose=2)

# Getting the model summary
model.summary()

# Saving the model
model.save("chatbot-model/")


# Reversing the word index
reverse_word_index = dict([(value, key)
                           for (key, value) in word_index.items()])


def decode_sentence(text):
    # Fancy Looking List Comprehension
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# print(decode_sentence(training_padded[0]))
# print(training_sentences[2])
# print(labels[2])

# e = model.layers[0]
# weights = e.get_weights()[0]
# print(weights.shape)  # shape: (vocab_size, embedding_dim)

# Making the model actually verbally speak
def answer(text):
    tts = gTTS(text=text, lang='en')
    filename = 'voice.mp3'
    tts.save(filename)
    playsound.playsound(filename)

# Search


def search_google(search_query):
    url = f"https://google-search3.p.rapidapi.com/api/v1/search/q={search_query}&num=100"

    headers = {
        'x-rapidapi-host': "google-search3.p.rapidapi.com",
        'x-rapidapi-key': "77ee9999bdmsh61f74ea752bc214p12f26cjsn3bc1f1725b22"
    }

    response = requests.request("GET", url, headers=headers)

    jsonated_response = response.json()
    title = jsonated_response['results'][0]['title']
    link = jsonated_response['results'][0]['link']
    webbrowser.open(link)
    return title

# Get time function


def get_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time


def get_ans(entry):
    msg = entry
    question_label['text'] = "Question: " + str(msg)
    sentence = []
    results = []

    sentence.append(msg)
    print(sentence)

    # Creating the responses for the stuff
    tag_0 = ["Hey how are you!", "Yo man"]
    tag_1 = ["Hey i am JARVIS your virtual assistant", "Hey my name is JARVIS"]
    tag_2 = ["I am 13 years old",
             "I am 13 years young", "13 years old man"]
    tag_3 = ["Goodbye", "See you soon", "Can't wait for another chat"]
    tag_4 = ["That is a bad thing to say.",
             "You should not use such bad language"]

    final_response = 0

    # Getting the predictions to display
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=max_length,
                           padding=padding_type, truncating=trunc_type)
    # print(model.predict(padded))
    prediction = model.predict(padded)

    print(prediction)

    # Getting the prediction and returning it mapped to the clsss names list defined above
    response = class_names[np.argmax(prediction)]
    print(np.argmax(prediction))
    print(response)

    if response == "Greeting":
        final_response = tag_0[random.randint(0, 1)]
        print(final_response)
        response_label['text'] = "Answer: " + str(final_response)
        answer(final_response)

    elif response == "Name":
        final_response = tag_1[random.randint(0, 1)]
        print(final_response)
        response_label['text'] = "Answer: " + str(final_response)
        answer(final_response)

    elif response == "Age":
        final_response = tag_2[random.randint(0, 2)]
        print(final_response)
        response_label['text'] = "Answer: " + str(final_response)
        answer(final_response)

    elif response == "Goodbye":
        final_response = tag_3[random.randint(0, 2)]
        print(final_response)
        response_label['text'] = "Answer: " + str(final_response)
        answer(final_response)

    elif response == "Profanity":
        final_response = tag_4[random.randint(0, 3)]
        print(final_response)
        response_label['text'] = "Answer: " + str(final_response)
        answer(final_response)

    elif response == "Search":
        final_response = search_google(msg)
        response_label['text'] = "Answer: " + str(final_response)
        search_answer = final_response
        answer(search_answer)

    elif response == "Time":
        final_response = get_time()
        response_label['text'] = "Answer: The time right now is: " + \
            str(final_response)
        time = final_response
        answer(time)


# Making the entry field
text_input = tk.Entry(root)
text_input.place(relwidth=0.75, relheight=0.15)

# Making the submit button
button = tk.Button(root, text='Send Message',
                   command=lambda: get_ans(text_input.get()))
button.place(relx=0.8, relwidth=0.18, relheight=0.15)


# Main Loop
root.title("JARVIS")
root.mainloop()
