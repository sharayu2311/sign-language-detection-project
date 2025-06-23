import os
import yaml
import numpy as np
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Load configuration from data.yaml
with open("/Users/sharayubodkhe/Documents/VSCode/project/Sign_language_data/data.yaml", "r") as file:
    config = yaml.safe_load(file)


# Extract paths and class labels
train_data_path = config["train"]
val_data_path = config["val"]
class_labels = config["names"]

# Verify directory structure
for label in class_labels:
    assert os.path.exists(os.path.join(train_data_path, label)), f"Missing directory: {os.path.join(train_data_path, label)}"
    assert os.path.exists(os.path.join(val_data_path, label)), f"Missing directory: {val_data_path}/{label}"

# Data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_data = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)
val_data = val_datagen.flow_from_directory(
    val_data_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Model definition
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_data,
    epochs=20,
    validation_data=val_data,
    callbacks=[early_stop, reduce_lr]
)

# Save the trained model
model.save('sign_language_model.h5')

# Evaluate the model
train_loss, train_acc = model.evaluate(train_data)
val_loss, val_acc = model.evaluate(val_data)

print(f"Training Accuracy: {train_acc:.2f}")
print(f"Validation Accuracy: {val_acc:.2f}")

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Real-time sign language detection
cap = cv2.VideoCapture(0)
model = load_model('sign_language_model.h5')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, (128, 128))
    input_image = resized_frame / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    predictions = model.predict(input_image)
    predicted_class = np.argmax(predictions)
    predicted_probability = predictions[0][predicted_class]

    cv2.putText(frame, f"Prediction: {class_labels[predicted_class]} ({predicted_probability:.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Sign Language Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
