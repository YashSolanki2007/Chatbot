# Just putting the dataset file for reference
# [
#     {
#         "sentence": "Hello What's up",
#         "tag": 0
#     },
#     {
#         "sentence": "Hi",
#         "tag": 0
#     },
#     {
#         "sentence": "Hello",
#         "tag": 0
#     },
#     {
#         "sentence": "What is your name",
#         "tag": 1
#     },
#     {
#         "sentence": "What's your Name",
#         "tag": 1
#     },
#     {
#         "sentence": "What is your age",
#         "tag": 2
#     },
#     {
#         "sentence": "How old are you",
#         "tag": 2
#     },
#     {
#         "sentence": "How many years young are you respected sir",
#         "tag": 2
#     },
#     {
#         "sentence": "Bye",
#         "tag": 3
#     },
#     {
#         "sentence": "Goodbye",
#         "tag": 3
#     },
#     {
#         "sentence": "See you soon",
#         "tag": 3
#     },
#     {
#         "sentence": "Bitch",
#         "tag": 4
#     },
#     {
#         "sentence": "Shut up you idiot",
#         "tag": 4
#     },
#     {
#         "sentence": "Why are you a bitch",
#         "tag": 4
#     },
#     {
#         "sentence": "Why are you a jerk",
#         "tag": 4
#     },
#     {
#         "sentence": "Why are you are a dissapointment",
#         "tag": 4
#     },
#     {
#         "sentence": "Siri is better",
#         "tag": 4
#     },
#     {
#         "sentence": "What is a cookie",
#         "tag": 5
#     },
#     {
#         "sentence": "What is",
#         "tag": 5
#     },
#     {
#         "sentence": "What is a social media platform",
#         "tag": 5
#     },
#     {
#         "sentence": "How to code",
#         "tag": 5
#     },
#     {
#         "sentence": "How to cook nachos",
#         "tag": 5
#     },
#     {
#         "sentence": "How to cook spinsi salsa",
#         "tag": 5
#     },
#     {
#         "sentence": "how to cook spinsi salsa",
#         "tag": 5
#     },
#     {
#         "sentence": "How to choose an IVY league university",
#         "tag": 5
#     },
#     {
#         "sentence": "How to choose a code editor",
#         "tag": 5
#     },
#     {
#         "sentence": "how to choose a code editor",
#         "tag": 5
#     },
#     {
#         "sentence": "What is IRCTC",
#         "tag": 5
#     },
#     {
#         "sentence": "What is the UN",
#         "tag": 5
#     },
#     {
#         "sentence": "Who owns google",
#         "tag": 5
#     },
#     {
#         "sentence": "who owns google",
#         "tag": 5
#     },
#     {
#         "sentence": "who is virat kohli",
#         "tag": 5
#     },
#     {
#         "sentence": "who is Bill Gates",
#         "tag": 5
#     },
#     {
#         "sentence": "When was Elon Musk born",
#         "tag": 5
#     },
#     {
#         "sentence": "What is the full form of USA",
#         "tag": 5
#     },
#     {
#         "sentence": "What is the time right now?",
#         "tag": 6
#     },
#     {
#         "sentence": "What is the time rn?",
#         "tag": 6
#     },
#     {
#         "sentence": "Wht is the time right now",
#         "tag": 6
#     },
#     {
#         "sentence": "What is the current time",
#         "tag": 6
#     }
# ]

# Imports
import json
import tensorflow as tf
import numpy as np
import pyaudio
import speech_recognition as sr
import pyttsx3
import wave
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

# Creating the responses for the stuff
tag_0 = ["Hey how are you!", "Yo man"]
tag_1 = ["Hey i am JARVIS your virtual assistant", "Hey my name is JARVIS"]
tag_2 = ["I am 13 years old",
         "I am 13 years young", "13 years old man"]
tag_3 = ["Goodbye", "See you soon", "Can't wait for another chat"]
tag_4 = ["That is a bad thing to say.",
         "You should not use such bad language"]


# Making the window
root = tk.Tk()
root.geometry("600x600")

# Making the response label
response_label = tk.Label(root, text='')
response_label.place(relx=0, rely=0.7, relwidth=1, relheight=0.15)

# Building the question label
question_label = tk.Label(root, text='')
question_label.place(relx=0, rely=0.6, relwidth=1, relheight=0.1)

# Important Variables
vocab_size = 800
embedding_dim = 17
max_length = 30
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
# Total size 39
training_size = 39
num_epochs = 300

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
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(800, activation='relu'),
    tf.keras.layers.Dense(700, activation='relu'),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(124, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(7, activation='sigmoid')
])

# Loading the model
model = keras.models.load_model("chatbot-model-2/")

# Compiling and fitting the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Fitting the model
# history = model.fit(training_padded, training_labels, epochs=num_epochs,
#                     validation_data=(testing_padded, testing_labels), verbose=2)

# Getting the model summary
model.summary()

# Saving the model
model.save("chatbot-model-2/")


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

# Getting the response function


def get_response(response, msg):
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
        final_response = tag_4[random.randint(0, 1)]
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


def get_ans(entry):
    global tag_0, tag_1, tag_2, tag_3, tag_4
    msg = entry
    question_label['text'] = "Question: " + str(msg)
    sentence = []
    results = []

    sentence.append(msg)
    print(sentence)

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

    get_response(response, msg)


def audio_speech():
    r = sr.Recognizer()

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # open the file
    with sr.AudioFile("output.wav") as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        msg = r.recognize_google(audio_data)
    print(msg)

    question_label['text'] = "Question: " + str(msg)
    sentence = []
    results = []

    sentence.append(msg)
    print(sentence)

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
    get_response(response, msg)


# Making the entry field
text_input = tk.Entry(root)
text_input.place(relwidth=0.75, relheight=0.15)

# Making the submit button
button = tk.Button(root, text='Send Message',
                   command=lambda: get_ans(text_input.get()))
button.place(relx=0.8, relwidth=0.18, relheight=0.15)

record_btn = tk.Button(root, text='Speak',
                       command=lambda: audio_speech())
record_btn.place(rely=0.3, relx=0.8, relwidth=0.18, relheight=0.15)


# Main Loop
root.title("JARVIS")
root.mainloop()
