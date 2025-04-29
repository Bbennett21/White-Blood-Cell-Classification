import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Define dataset path
dataset_path = "./bloodcells_dataset"
batch_size = 32
img_size = (360, 363)  # Original size
target_size = (256, 256)

# Load original images
dataset = tf.keras.preprocessing.image_dataset_from_directory(dataset_path, image_size=img_size, batch_size=batch_size)


# Resize images using bilinear interpolation
def resize_images(image, label):
    resized_image = tf.image.resize(image, target_size, method='bilinear')
    return resized_image, label

# Apply the resize function to the dataset
resized_dataset = dataset.map(resize_images)


# Normalize pixel values to [0, 1] 
def normalize_images(image, label):
    normalized_image = tf.cast(image, tf.float32) / 255.0
    return normalized_image, label

# Apply the normalize function to the resized dataset
processed_dataset = resized_dataset.map(normalize_images)

#Split the dataset into testing, validation, and training data
dataset_size = tf.data.experimental.cardinality(processed_dataset).numpy()
training_size = int(0.8 * dataset_size)
validation_size = int(0.1 * dataset_size)
testing_size = dataset_size - training_size - validation_size

training_dataset = processed_dataset.take(training_size)
remaining_dataset = processed_dataset.skip(training_size)
validation_dataset = remaining_dataset.take(validation_size)
testing_dataset = remaining_dataset.skip(validation_size)


# Get the number of classes from your dataset
num_classes = len(dataset.class_names)


#CNN Model
model = Sequential([
    # Input layer
    Input(shape=(256, 256, 3)),
    
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Third Convolutional Block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Fourth Convolutional Block
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Flatten and Dense layers
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the CNN
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])


# Train the model
history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=15,
    callbacks=[
        # Stop early if there are 3 epochs in a row with no improvement
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        # Save the model with the best perfromance 
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_model.keras',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
)

plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(testing_dataset)
print(f"Test accuracy: {test_accuracy:.4f}")

# Evaluate the model on all datasets
print("Evaluating the model on all datasets...")

# Evaluate on training dataset
train_loss, train_accuracy = model.evaluate(training_dataset, verbose=1)
print("Train Loss: ", train_loss)
print("Train Accuracy: ", train_accuracy)
print('-' * 20)

# Evaluate on validation dataset
valid_loss, valid_accuracy = model.evaluate(validation_dataset, verbose=1)
print("Validation Loss: ", valid_loss)
print("Validation Accuracy: ", valid_accuracy)
print('-' * 20)

# Evaluate on test dataset
test_loss, test_accuracy = model.evaluate(testing_dataset, verbose=1)
print("Test Loss: ", test_loss)
print("Test Accuracy: ", test_accuracy)

# Generate confusion matrix


# Function to get predictions and true labels from a dataset
def get_predictions(dataset):
    y_pred = []
    y_true = []
    
    for images, labels in dataset:
        predictions = model.predict(images)
        pred_classes = np.argmax(predictions, axis=1)
        
        y_pred.extend(pred_classes)
        y_true.extend(labels.numpy())
    
    return np.array(y_true), np.array(y_pred)

# Get predictions for test dataset
y_true, y_pred = get_predictions(testing_dataset)

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=dataset.class_names, yticklabels=dataset.class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Print classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=dataset.class_names))