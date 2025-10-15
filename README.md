# Next-Word-Prediction-Project


## 📌 Overview
The **Next Word Prediction Project** is a Natural Language Processing (NLP) project that uses **Long Short-Term Memory (LSTM)** neural networks to predict the most likely next word in a given sequence of text.  
This type of model is widely used in applications like autocomplete, chatbots, predictive typing, and AI-driven assistants.

---

## ⚙️ Technologies Used
- **Python 3**
- **TensorFlow / Keras** – for building and training the LSTM model
- **NumPy & Pandas** – for data preprocessing
- **Matplotlib / Seaborn** – for visualization
- **Natural Language Toolkit (NLTK)** – for text cleaning and tokenization

---

## 🧠 What is LSTM (Long Short-Term Memory)?
**LSTM** is a special type of Recurrent Neural Network (**RNN**) designed to remember long-term dependencies in sequences.  
In traditional RNNs, the network struggles with the **vanishing gradient problem**, making it hard to learn from words that appear far back in the sequence.  
LSTMs solve this by using a **memory cell** that can store information for long periods.

---

### 🔑 Key Components of LSTM
1. **Cell State** – The "memory" of the network, carrying important information across time steps.  
2. **Forget Gate** – Decides which information should be discarded from memory.  
3. **Input Gate** – Decides which new information should be added to memory.  
4. **Output Gate** – Decides what part of the memory should be output at the current step.  

This structure helps LSTMs **learn context over long sequences**, which is crucial in language tasks like Next Word Prediction.

---

### 📘 Why LSTM for Next Word Prediction?
- **Understands context** – Learns how words depend on each other in a sentence.  
- **Handles long sequences** – Can remember information from earlier words.  
- **Better than vanilla RNNs** – Solves vanishing gradient issue, making training more stable.  

Example:  
Input: `"I am going to the"`  
- A simple model might predict `"is"`, `"and"`, etc.  
- An **LSTM model**, having learned context, predicts `"market"`, `"school"`, or `"park"` (more meaningful words).  

---

### 🖼️ LSTM Working (Simplified)
Input Sequence → Embedding → LSTM Layers → Dense Layer → Predicted Next Word


---

✅ In this project, we used **LSTM** because it can capture the **sequence and meaning of words**, making predictions more accurate than simple statistical models.


---

## 🚀 How It Works
1. **Dataset Preparation**  
   - The text corpus is cleaned, tokenized, and converted into sequences of words.  
   - Example: `"I love to play"` → Input: `"I love to"` → Output: `"play"`

2. **Model Architecture (LSTM)**  
   - Embedding Layer (to convert words into dense vectors)  
   - LSTM Layers (to learn sequential dependencies)  
   - Dense Layer with Softmax Activation (to predict the next word)

3. **Training**  
   - The model is trained on sequences to minimize categorical cross-entropy loss.  
   - Optimizer: **Adam**  

4. **Prediction**  
   - The trained model takes a sequence of words and predicts the next most probable word.  
   - Example:  
     Input: `"The sun is"` → Predicted Output: `"shining"`

---

## 📊 Importance of Next Word Prediction
- Enhances **user experience** in messaging apps and search engines.  
- Powers **chatbots** and **virtual assistants** like Alexa, Google Assistant, and Siri.  
- Helps in **language modeling** and **text generation** tasks.  
- Reduces typing effort through **autocomplete** functionality.

---

## 📂 Project Structure

📦 Next-Word-Prediction-Project  
 ┣ 📜 next_word_prediction.ipynb   # Colab File  
 ┣ 📜 data.txt                     # Dataset  
 ┗ 📜 README.md                    # Project Documentation  


---
