import pickle
import tensorflow as tf

from preprocess import preprocess_input

# Load model
model = tf.keras.models.load_model("../model/sarcasm_model.keras")

# Load tokenizer
with open("../tokenizer/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def predict(text):
    padded = preprocess_input(text, tokenizer)
    pred = model.predict(padded)[0][0]
    return "Sarcastic" if pred > 0.5 else "Not Sarcastic"

if __name__ == "__main__":
    while True:
        text = input("Enter a sentence (or type 'exit'): ")
        if text.lower() == "exit":
            break
        print(predict(text))
