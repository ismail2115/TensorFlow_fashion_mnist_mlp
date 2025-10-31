# ================================================
#  Fashion-MNIST Classification (Student Version)
#  Author: [Your Name]
#  Course: AI / Deep Learning Assignment
# ================================================

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------
# 1. Reproducibility & Setup
# ------------------------------------------------
RANDOM_SEED = 123
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ------------------------------------------------
# 2. Load & Prepare Dataset
# ------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

NUM_CLASSES = 10
print(f"✅ Dataset loaded successfully. Train: {x_train.shape}, Test: {x_test.shape}")

# ------------------------------------------------
# 3. Data Augmentation
# ------------------------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomTranslation(0.05, 0.05),
], name="augmentation")

# ------------------------------------------------
# 4. CNN Model Definition
# ------------------------------------------------
def build_cnn(input_shape=(28, 28, 1), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.35)(x)

    # Fully connected
    x = layers.Flatten()(x)
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="fashion_mnist_cnn_final")
    return model


model = build_cnn()
model.summary()

# ------------------------------------------------
# 5. Compile Model
# ------------------------------------------------
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ------------------------------------------------
# 6. Callbacks Setup
# ------------------------------------------------
os.makedirs("checkpoints", exist_ok=True)

checkpoint = ModelCheckpoint(
    filepath="checkpoints/best_model.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

csv_logger = CSVLogger("training_log.csv", append=False)

callbacks_list = [checkpoint, early_stop, reduce_lr, csv_logger]

# ------------------------------------------------
# 7. Train the Model
# ------------------------------------------------
history = model.fit(
    x_train, y_train,
    epochs=25,
    batch_size=64,
    validation_split=0.15,
    callbacks=callbacks_list,
    verbose=2
)

# Save history to CSV
pd.DataFrame(history.history).to_csv("history.csv", index=False)
print("✅ Training history saved to history.csv")

# ------------------------------------------------
# 8. Evaluate on Test Set
# ------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n📊 Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# ------------------------------------------------
# 9. Plot Accuracy & Loss Curves
# ------------------------------------------------
plt.figure(figsize=(10, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend(); plt.title("Model Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend(); plt.title("Model Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")

plt.tight_layout()
plt.savefig("training_curves_student.png")
plt.show()

# ------------------------------------------------
# 10. Save Final Model
# ------------------------------------------------
model.save("fashion_mnist_cnn_final.h5")
print("💾 Final model saved as fashion_mnist_cnn_final.h5")

# ------------------------------------------------
# 11. Reload and Test the Best Checkpointed Model
# ------------------------------------------------
from tensorflow.keras.models import load_model

best_model = load_model("checkpoints/best_model.h5")
best_loss, best_acc = best_model.evaluate(x_test, y_test, verbose=0)

print(f"\n🏆 Best checkpointed model accuracy: {best_acc:.4f}, loss: {best_loss:.4f}")

# ------------------------------------------------
# 12. Make Example Predictions
# ------------------------------------------------
predictions = best_model.predict(x_test[:10])
predicted_labels = np.argmax(predictions, axis=1)

print("\n🔍 First 10 Predictions:", predicted_labels)
print("✅ All files and models have been generated successfully.")
