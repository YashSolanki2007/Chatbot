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


# Important Variables
vocab_size = 500
embedding_dim = 16
max_length = 10
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 8
num_epochs = 100

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
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(124, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(4, activation='sigmoid')
])

# Loading the model
# model = keras.models.load_model("chatbot-model/")

# Compiling and fitting the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Getting the model summary
model.summary()

# Fitting the model
history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels), verbose=2)

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

# Doing all the predictions stuff

while True:
    msg = input(">")
    sentence = []

    sentence.append(msg)
    print(sentence)

    # Creating the responses for the stuff
    tag_0 = ["Hey how are you!", "Yo man"]
    tag_1 = ["Hey i am TARS your virtual assistant", "Hey my name is TARS"]
    tag_2 = ["I am 13 years old", "I am 13 years young", "13 years old man"]
    tag_3 = ["Goodbye", "See you soon", "Can't wait for another chat"]

    random_index_0 = random.randint(0, 1)
    random_index_1 = random.randint(0, 1)
    random_index_2 = random.randint(0, 2)
    random_index_3 = random.randint(0, 2)

    final_response = 0

    # Getting the predictions to display
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=max_length,
                           padding=padding_type, truncating=trunc_type)
    # print(model.predict(padded))
    prediction = model.predict(padded)

    for i in range(len(prediction[0])):
        print(round(prediction[0][i]))

    if round(prediction[0][0]) == 1.0:
        final_response = tag_0[random_index_0]
        print(final_response)

    elif round(prediction[0][1]) == 1.0:
        final_response = tag_1[random_index_1]
        print(final_response)

    elif round(prediction[0][2]) == 1.0:
        final_response = tag_2[random_index_2]
        print(final_response)

    elif round(prediction[0][3]) == 1.0:
        final_response = tag_3[random_index_3]
        print(final_response)

    else:
        print("I didn't quite get that please try again")


# msg = input(">")
# sentence = []

# sentence.append(msg)
# print(sentence)

# # Creating the responses for the stuff
# tag_0 = ["Hey how are you!", "Yo man"]
# tag_1 = ["Hey i am TARS your virtual assistant", "Hey my name is TARS"]
# tag_2 = ["I am 13 years old", "I am 13 years young", "13 years old, man"]
# tag_3 = ["Goodbye", "See you soon", "Can't wait for another chat"]

# random_index_0 = random.randint(0, 1)
# random_index_1 = random.randint(0, 1)
# random_index_2 = random.randint(0, 2)
# random_index_3 = random.randint(0, 2)

# final_response = 0

# # Getting the predictions to display
# sequences = tokenizer.texts_to_sequences(sentence)
# padded = pad_sequences(sequences, maxlen=max_length,
#                        padding=padding_type, truncating=trunc_type)
# # print(model.predict(padded))
# prediction = model.predict(padded)

# for i in range(len(prediction[0])):
#     print(round(prediction[0][i]))

# if round(prediction[0][0]) == 1.0:
#     final_response = tag_0[random_index_0]
#     print(final_response)

# elif round(prediction[0][1]) == 1.0:
#     final_response = tag_1[random_index_1]
#     print(final_response)

# elif round(prediction[0][2]) == 1.0:
#     final_response = tag_2[random_index_2]
#     print(final_response)

# elif round(prediction[0][3]) == 1.0:
#     final_response = tag_3[random_index_3]
#     print(final_response)

# else:
#     print("I didn't quite get that please try again")
