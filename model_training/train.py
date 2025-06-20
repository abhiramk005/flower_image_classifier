import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os

# === Paths ===
data_dir = "flowers"
model_path = "../model/flower_model.h5"

# === Data Augmentation & Preprocessing ===
datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1. / 255,
    horizontal_flip=True,
    zoom_range=0.2
)

img_size = (224, 224)
batch_size = 32

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Important for evaluation
)

# === Load Pretrained Model ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# === Build Model ===
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === Train ===
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# === Save Model ===
os.makedirs("model", exist_ok=True)
model.save(model_path)
print("✅ Model saved to", model_path)


# === Evaluation ===
def evaluate_model(model, generator):
    # Get true labels and predictions
    y_true = generator.classes
    y_pred = model.predict(generator)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Get class names
    class_names = list(generator.class_indices.keys())

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.xticks(rotation=45)
    plt.title("Confusion Matrix")

    plt.show()

    return y_true, y_pred_classes


# Evaluate on validation set
print("\nEvaluating on Validation Set...")
y_true, y_pred = evaluate_model(model, val_generator)

# Optional: Evaluate on training set
print("\nEvaluating on Training Set...")
train_generator_for_eval = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=False  # Important for evaluation
)
evaluate_model(model, train_generator_for_eval)