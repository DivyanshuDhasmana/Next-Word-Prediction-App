import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# ----------------- Load Tokenizer -----------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
seq_length = 100  # same as your training sequence length

# ----------------- Build Model -----------------
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=seq_length))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(150))
model.add(Dense(vocab_size, activation='softmax'))

model.build(input_shape=(None, seq_length))

# ----------------- Load Weights -----------------
model.load_weights("my_model.weights.h5")

# ----------------- Reverse Mapping -----------------
index_word = {index: word for word, index in tokenizer.word_index.items()}

# ----------------- Prediction Function -----------------
def generate_text(seed_text, next_words=10):
    result = seed_text
    for _ in range(next_words):
        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([seed_text])[0]
        # Pad sequence
        sequence = pad_sequences([sequence], maxlen=seq_length, padding='pre')
        # Predict next word
        predicted_probs = model.predict(sequence, verbose=0)[0]
        predicted_index = np.argmax(predicted_probs)
        predicted_word = index_word.get(predicted_index, '')
        # Append predicted word
        result += ' ' + predicted_word
        # Update seed_text
        seed_text += ' ' + predicted_word
        seed_text = ' '.join(seed_text.split()[1:])  # shift window
    return result

# ----------------- Streamlit UI -----------------
st.title("Next Word Prediction App")
st.write("Enter some text and let the model predict the next words!")

user_input = st.text_input("Your Text Here", "Once upon a time")
num_words = st.slider("Number of Words to Predict", 1, 20, 10)

if st.button("Predict"):
    output = generate_text(user_input, next_words=num_words)
    st.subheader("Predicted Text:")
    st.write(output)
