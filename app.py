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

# Vocabulary size (needed to recreate the model)
vocab_size = len(tokenizer.word_index) + 1

# ----------------- Recreate Model Architecture -----------------
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=56))
model.add(LSTM(128))
model.add(Dense(vocab_size, activation='softmax'))

# ----------------- Load Weights -----------------
model.load_weights("my_model.weights.h5")  # correct filename

# ----------------- Prediction Function -----------------
def generate_text(seed_text, next_words=10):
    text = seed_text
    for _ in range(next_words):
        token_text = tokenizer.texts_to_sequences([text])[0]
        padded_token_text = pad_sequences([token_text], maxlen=56, padding='pre')
        predictions = model.predict(padded_token_text, verbose=0)
        predicted_index = np.argmax(predictions)
        predicted_word = index_word.get(predicted_index, "")
        text += " " + predicted_word
    return text

# ----------------- Streamlit UI -----------------
st.title("🔮 Next Word Prediction App")
st.write("Type a sentence and the model will generate the next words.")

seed_text = st.text_input("Enter your starting sentence:", "what is the fee")
num_words = st.slider("How many words to generate?", 1, 20, 10)

if st.button("Generate"):
    result = generate_text(seed_text, next_words=num_words)
    st.success(f"Generated Text:\n\n{result}")
