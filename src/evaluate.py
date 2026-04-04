import pickle
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from data_loader import load_data

# Load data
sentences, labels = load_data("../data/dataset.json")

# Load tokenizer
with open("../tokenizer/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Preprocess using SAME tokenizer
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=25, padding='post')

# Split SAME as training
X_train, X_test, y_train, y_test = train_test_split(
    padded, labels, test_size=0.2, random_state=42
)

# Load model
model = tf.keras.models.load_model("../model/sarcasm_model.keras")

# Predict
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))