# =========================================================
# LSTM Text Generation using Shakespeare Dataset
# =========================================================

import re
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================================================
# 1. DATASET LOADING & PREPROCESSING
# =========================================================

# Load text file
with open("shakespeare.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Convert to lowercase
text = text.lower()

# Remove punctuation and special characters
text = re.sub(r"[^a-z\s]", "", text)

print("Dataset loaded successfully")
print("Total characters:", len(text))

# =========================================================
# 2. TOKENIZATION & SEQUENCE CREATION
# =========================================================

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1
print("Total unique words:", total_words)

# Create input sequences
input_sequences = []

for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i + 1])

# Padding sequences
max_seq_len = max(len(seq) for seq in input_sequences)

input_sequences = pad_sequences(
    input_sequences,
    maxlen=max_seq_len,
    padding="pre"
)

# Split input and output
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# One-hot encode output
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

print("Input shape:", X.shape)
print("Output shape:", y.shape)

# =========================================================
# 3. MODEL DESIGN (LSTM)
# =========================================================

model = Sequential([
    Embedding(total_words, 100, input_length=max_seq_len - 1),
    LSTM(150, return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dense(total_words, activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# =========================================================
# 4. MODEL TRAINING
# =========================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

model.fit(
    X,
    y,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop]
)

# =========================================================
# 5. TEXT GENERATION FUNCTION
# =========================================================

def generate_text(seed_text, next_words=30):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list],
            maxlen=max_seq_len - 1,
            padding="pre"
        )

        predicted_index = np.argmax(
            model.predict(token_list, verbose=0),
            axis=-1
        )[0]

        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                seed_text += " " + word
                break

    return seed_text

# =========================================================
# 6. SAMPLE TEXT GENERATION OUTPUT
# =========================================================

print("\n--- Generated Text Samples ---\n")

seed1 = "to be or not to be"
print("Seed:", seed1)
print(generate_text(seed1, 25))
print("\n")

seed2 = "love is"
print("Seed:", seed2)
print(generate_text(seed2, 25))
print("\n")

seed3 = "the king"
print("Seed:", seed3)
print(generate_text(seed3, 25))
