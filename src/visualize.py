import pickle
import matplotlib.pyplot as plt
import os

# Ensure plots folder exists
os.makedirs("../plots", exist_ok=True)

# Load history
with open("../model/history.pkl", "rb") as f:
    history = pickle.load(f)


# Accuracy Plot
plt.figure()
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Save
plt.savefig("../plots/accuracy.png")

# Show
plt.show()


# Loss Plot
plt.figure()
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Save
plt.savefig("../plots/loss.png")

# Show
plt.show()
