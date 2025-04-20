import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

import os

dataset_path = r"C:\Users\Admin\Desktop\Chest X-ray Detection\chest_xray"

# Define paths for train, test, and validation sets
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")
val_path = os.path.join(dataset_path, "val")

# Count images in each category
for category in ["train", "test", "val"]:
    for label in ["NORMAL", "PNEUMONIA"]:
        folder_path = os.path.join(dataset_path, category, label)
        print(f"{category} - {label}: {len(os.listdir(folder_path))} images")

import tensorflow as tf

batch_size = 32
img_size = (180, 180)

# Load training data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)

# Load test data
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    image_size=img_size,
    batch_size=batch_size
)

# Load validation data
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_path,
    image_size=img_size,
    batch_size=batch_size
)

# Display dataset structure
print(train_ds.class_names)  # Should print ['NORMAL', 'PNEUMONIA']


from tensorflow.keras import layers

normalization_layer = layers.Rescaling(1./255)

# Apply augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2)
])

# Apply preprocessing to the dataset
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

from tensorflow import keras

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(180, 180, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # Binary classification
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()
epochs = 10

history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
model.save("pneumonia_detector.h5")