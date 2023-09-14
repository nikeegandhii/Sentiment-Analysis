# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the IMDb movie reviews dataset (replace with your dataset path)
df = pd.read_csv("imdb_reviews.csv")

# Data preprocessing
reviews = df["review"].values
sentiments = df["sentiment"].map({"positive": 1, "negative": 0}).values

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(
    reviews, sentiments, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Tokenize and pad sequences
max_words = 10000
max_sequence_length = 200

# Initialize a tokenizer to convert text to numerical sequences
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

# Convert text data to sequences of integers
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_val_sequences = tokenizer.texts_to_sequences(X_val)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure consistent input length for the model
X_train_padded = pad_sequences(
    X_train_sequences, maxlen=max_sequence_length, padding="post"
)
X_val_padded = pad_sequences(
    X_val_sequences, maxlen=max_sequence_length, padding="post"
)
X_test_padded = pad_sequences(
    X_test_sequences, maxlen=max_sequence_length, padding="post"
)

# Build and train a sentiment analysis model
model = keras.Sequential(
    [
        keras.layers.Embedding(
            input_dim=max_words, output_dim=128, input_length=max_sequence_length
        ),
        keras.layers.LSTM(64),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(
    X_train_padded,
    y_train,
    epochs=5,
    batch_size=128,
    validation_data=(X_val_padded, y_val),
)

# Evaluate the model
y_pred = (model.predict(X_test_padded) > 0.5).astype(int)

# Calculate accuracy and other metrics
accuracy = accuracy_score(y_test, y_pred)
confusion_matrix_result = confusion_matrix(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(confusion_matrix_result)
print("Classification Report:")
print(classification_report_result)

# Plot training and validation loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()