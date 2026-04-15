# 🤖 Sarcasm Detection using Deep Learning (Bi-LSTM)

A complete **Natural Language Processing (NLP)** project that detects sarcasm in text using a **Bidirectional LSTM (Bi-LSTM)** model.
The system is built with a clean, modular architecture and evaluated on multiple datasets to ensure **real-world generalization**.

---

## 🚀 Project Overview

Sarcasm detection is a challenging NLP task because sarcastic sentences often appear positive on the surface but convey negative intent.

This project builds an end-to-end pipeline that:

* Processes raw text data
* Converts it into numerical representations
* Trains a deep learning model
* Evaluates performance using multiple metrics
* Tests generalization on unseen datasets

---

## 🧠 Problem Statement

Given a sentence:

```
"Oh great, another assignment"
```

The model predicts:

```
Sarcastic 😏
```

---

## 🏗️ Project Structure

```
sarcasm-detector/
│
├── data/
│   ├── dataset.json
│   └── new_dataset.json
│
├── model/
│   ├── sarcasm_model.keras
│   └── history.pkl
│
├── tokenizer/
│   └── tokenizer.pkl
│
├── plots/
│   ├── accuracy.png
│   └── loss.png
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── test_new_data.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone <your-repo-url>
cd sarcasm-detector
```

### 2. Create virtual environment

```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## 📊 Dataset

This project uses the **News Headlines Sarcasm Dataset** from Kaggle.

* Format: JSON (line-by-line)
* Fields:

  * `headline` → text
  * `is_sarcastic` → label (0 or 1)

Download it from:
[https://www.kaggle.com/search?q=News+Headlines+Sarcasm+Dataset]

Note: Dataset is subject to its own license.

---

## 🔄 Pipeline

### 1. Data Loading

* Reads JSON dataset
* Extracts text and labels

### 2. Preprocessing

* Tokenization (word → integer mapping)
* Padding (fixed sequence length)

### 3. Model Architecture

* Embedding Layer
* Bidirectional LSTM (64 units)
* Dropout (regularization)
* Bidirectional LSTM (32 units)
* Dense layers
* Sigmoid output (binary classification)

---

## 🧠 Model Summary

* Total Parameters: ~748K
* Type: Deep Learning (Bi-LSTM)
* Loss: Binary Crossentropy
* Optimizer: Adam

---

## 🏋️ Training

Run:

```
cd src
python train.py
```

Features:

* Early stopping (prevents overfitting)
* Model saving (`.keras` format)
* Tokenizer persistence
* Training history saved for visualization

---

## 🔮 Prediction

Run:

```
python predict.py
```

Example:

```
Enter a sentence: Oh great, another exam
Sarcastic 😏
```

---

## 📈 Evaluation

Run:

```
python evaluate.py
```

Metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## 📊 Visualization

Run:

```
python visualize.py
```

Generates:

* Accuracy vs Epoch graph
* Loss vs Epoch graph

Saved in:

```
plots/
```

---

## 🔍 Cross-Dataset Testing

Run:

```
python test_new_data.py
```

This evaluates the model on a **completely different dataset** to test generalization.

---

## 📊 Results

### ✅ Primary Dataset

* Accuracy: ~85–87%
* F1 Score: ~0.86

### ✅ Cross-Dataset Performance

* Accuracy: ~91%
* F1 Score: ~0.90

---

## 🧠 Key Insights

* The model generalizes well across datasets
* Slight difficulty in detecting subtle sarcasm
* Performance depends on dataset style and distribution

---

## ⚠️ Challenges

* Sarcasm is context-dependent
* Limited understanding of real-world tone
* Domain mismatch (news vs conversational text)

---

## 🔥 Future Improvements

* Replace LSTM with Transformer models (BERT)
* Add Streamlit UI for interactive predictions
* Deploy as a web application
* Fine-tune on conversational sarcasm datasets

---

## 🧑‍💻 Technologies Used

* Python
* TensorFlow / Keras
* NumPy / Pandas
* Scikit-learn
* Matplotlib

---

## 🏁 Conclusion

This project demonstrates:

* End-to-end NLP pipeline design
* Deep learning model implementation
* Proper evaluation and validation
* Real-world testing using multiple datasets

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!

---
