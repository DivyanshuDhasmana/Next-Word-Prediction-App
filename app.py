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
        sequence = tokenizer.texts_to_sequences([seed_text])[0]
        sequence = pad_sequences([sequence], maxlen=seq_length, padding='pre')
        predicted_probs = model.predict(sequence, verbose=0)[0]
        predicted_index = np.argmax(predicted_probs)
        predicted_word = index_word.get(predicted_index, '')
        result += ' ' + predicted_word
        seed_text += ' ' + predicted_word
        seed_text = ' '.join(seed_text.split()[1:])
    return result

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Next Word Predictor", layout="wide")

# Top-right name
st.markdown(
    """
    <div style="
        position: absolute;
        top: 10px;
        right: 20px;
        font-size: 20px;
        font-weight: bold;
        color: #1f77b4;
        text-decoration: underline;
        z-index: 1000;
    ">
        Divyashu Dhasmana
    </div>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Next Word Predictor (LSTM)")
st.markdown("Type a few words, and the model will generate the next words for you.")

# User input section
user_input = st.text_area("Enter your sentence:", height=100, placeholder="e.g., Once upon a time")
num_words = st.slider("Number of words to predict:", min_value=1, max_value=20, value=10)

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        output = generate_text(user_input, next_words=num_words)
        st.success("Predicted Text:")
        st.write(output)
