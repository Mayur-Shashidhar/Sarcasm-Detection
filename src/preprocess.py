from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess(sentences, vocab_size=10000, max_length=25):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)

    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')

    return padded, tokenizer

def preprocess_input(text, tokenizer, max_length=25):
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=max_length, padding='post')