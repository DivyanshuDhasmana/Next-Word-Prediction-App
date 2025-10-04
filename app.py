import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# ----------------- Load Tokenizer -----------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Build reverse mapping (index → word)
index_word = {index: word for word, index in tokenizer.word_index.items()}

vocab_size = len(tokenizer.word_index) + 1  # total vocabulary size
seq_length = 10  # replace with the sequence length you used during training

# ----------------- Build Model -----------------
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=seq_length))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation='softmax'))

# ----------------- Build the model before loading weights -----------------
model.build(input_shape=(None, seq_length))

# ----------------- Load Weights -----------------
model.load_weights("my_model.weights.h5")

# ----------------- Prediction Function -----------------
def generate_text(seed_text, next_words=10):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=seq_length, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]
        output_word = index_word.get(predicted_index, "")
        seed_text += " " + output_word
    return seed_text

# ----------------- Streamlit Interface -----------------
st.title("Next Word Prediction App")
seed_text = st.text_input("Enter seed text:")
next_words = st.number_input("Number of words to predict:", min_value=1, max_value=50, value=10)

if st.button("Generate"):
    result = generate_text(seed_text, next_words)
    st.write(result)
