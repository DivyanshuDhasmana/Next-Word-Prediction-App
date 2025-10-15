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
def predict_next_words(seed_text, top_n=5):
    sequence = tokenizer.texts_to_sequences([seed_text])[0]
    sequence = pad_sequences([sequence], maxlen=seq_length, padding='pre')
    
    predicted_probs = model.predict(sequence, verbose=0)[0]
    
    top_indices = predicted_probs.argsort()[-top_n:][::-1]
    top_words = [index_word.get(i, "") for i in top_indices]
    
    return top_words

# ----------------- Streamlit UI -----------------
st.title("Next Word Prediction App")
st.write("Enter a text and see the top predicted next words!")

user_input = st.text_input("Your Text Here", "Once upon a time")
top_n = st.slider("Number of Predictions to Show", 1, 10, 5)

if st.button("Predict"):
    predictions = predict_next_words(user_input, top_n=top_n)
    st.subheader("Top Predicted Next Words:")
    st.write(", ".join(predictions))
