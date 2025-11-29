# Ag-News-Category-Text-Classification

---

## 📌 Project Overview

This project builds a deep learning model to classify news articles from the **AG News dataset** into four categories:

1. **World**
2. **Sports**
3. **Business**
4. **Sci/Tech**

The solution uses a complete Natural Language Processing (NLP) workflow including text preprocessing, word embeddings using Word2Vec, and an LSTM-based neural network for classification.

---

## 🎯 Objective

To develop a robust text classification model that:

- Understands and processes raw news text  
- Learns semantic meaning of words using Word2Vec  
- Accurately predicts the correct news category  
- Demonstrates an end-to-end deep learning NLP pipeline  

---

## 🧠 Key Features

- Custom Word2Vec embeddings trained on the dataset  
- Tokenization & padding of sequences  
- Embedding matrix creation for deep learning  
- LSTM-based text classifier built using Keras  
- Multi-class prediction (4 categories)  
- Supports reproducible training and evaluation  

---

## 🛠 Tech Stack

- **Python**
- **pandas, NumPy**
- **gensim** (Word2Vec)
- **TensorFlow / Keras**
- **NLTK**
- **Matplotlib**
- **Scikit-learn**

---

## ⚙️ NLP Pipeline

### 1️⃣ Data Loading  
Loaded the **AG_news_Dataset.csv** with title + description fields.

### 2️⃣ Text Preprocessing  
- Lowercasing  
- Removing punctuation  
- Removing numbers  
- Tokenization  
- Stopword removal  

### 3️⃣ Word2Vec Embedding  
- Trained Word2Vec using `gensim`  
- Created vocabulary  
- Built **embedding matrix** mapping each token → vector  

### 4️⃣ Sequence Preparation  
- Tokenizer fitted on text  
- Text converted into sequences  
- Padded to fixed length  

### 5️⃣ Model Architecture (LSTM)  
A Keras Sequential model:

- Embedding layer (initialized with Word2Vec)  
- LSTM layer  
- Dense + ReLU  
- Dropout  
- Softmax output (4 classes)  

### 6️⃣ Training & Evaluation  
- Train-test split  
- Model compiled with `adam` + `categorical_crossentropy`  
- Evaluated accuracy, loss  

---

## 📁 Project Structure

AG-News-Classification/

│── Ag_news_category_text_classification_task.ipynb

│── Ag_news_Dataset.csv

│── README.md

│── requirements.txt

--- 

## ▶️ How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Open the project notebook
   
jupyter notebook "Ag_news_category_text_classification_task.ipynb"

### 3. Run all cells

The notebook trains the Word2Vec model, builds the embedding matrix, trains LSTM, and outputs predictions.

---

🚀 **Future Improvements**

Use pre-trained embeddings like GloVe / FastText

Add BiLSTM or GRU layers

Deploy as a REST API

Convert model to TensorFlow Lite

Add prediction dashboard using Streamlit

---

👤 **Author**

**Srikanth Edigi**

📧 **Email**: srikanthgoud9515@gmail.com

🔗 **LinkedIn**: http://www.linkedin.com/in/srikanth-edigi-4739b125b
