import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import plot_accuracy, plot_confusion_matrix

# -------------------
# Load Dataset
# -------------------
X = np.load("images_mls.npy")  # Shape: (1342, 128, 128, 3)
labels = pd.read_csv("Labels_mls.csv")

y = labels["Label"].values

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize
X_train, X_test = X_train / 255.0, X_test / 255.0

# -------------------
# Build Model (Transfer Learning with VGG16)
# -------------------
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
for layer in base_model.layers:
    layer.trainable = False  # Freeze base layers

x = Flatten()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(len(np.unique(y)), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------
# Train Model
# -------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32,
    verbose=1
)

# -------------------
# Evaluate Model
# -------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {acc:.4f}")

# Save model
model.save("monkey_species_classifier.h5")

# -------------------
# Generate Plots
# -------------------
plot_accuracy(history, "accuracy_plot.png")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

class_names = encoder.classes_
plot_confusion_matrix(y_test, y_pred_classes, class_names, "confusion_matrix.png")
