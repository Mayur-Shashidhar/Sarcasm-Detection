import os
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf

from data_loader import load_data
from preprocess import preprocess
from model import build_model

# Create folders if not exist
os.makedirs("../tokenizer", exist_ok=True)
os.makedirs("../model", exist_ok=True)

# Load data
sentences, labels = load_data("../data/dataset.json")

# Preprocess
padded, tokenizer = preprocess(sentences)

# Save tokenizer
with open("../tokenizer/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    padded, labels, test_size=0.2, random_state=42
)

# Build model
model = build_model()
model.build(input_shape=(None, 25))
model.summary()

# Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Save history
with open("../model/history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# Save model
model.save("../model/sarcasm_model.keras")

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", acc)