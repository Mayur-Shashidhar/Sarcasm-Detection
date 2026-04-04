import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix


# Load NEW dataset
df = pd.read_json("../data/new_dataset.json", lines=True)

sentences = df['headline'].values
labels = df['is_sarcastic'].values


# Load tokenizer
with open("../tokenizer/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# Preprocess
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=25, padding='post')


# Load model
model = tf.keras.models.load_model("../model/sarcasm_model.keras")


# Predict
y_pred = model.predict(padded)
y_pred = (y_pred > 0.5).astype(int)


# Results
print("\nClassification Report:")
print(classification_report(labels, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(labels, y_pred))