import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load FER-2013 dataset
train_dir = "C:\\Users\\student\\Desktop\\development project\\input\\train"
test_dir = "C:\\Users\\student\\Desktop\\development project\\input\\test"

# Data generators with augmentation
train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    rescale=1./255,
    validation_split=0.2
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training'
)

validation_generator = validation_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'
)

# Define the visualize_samples function before calling it
def visualize_samples(generator):
    x_batch, y_batch = next(generator)
    plt.figure(figsize=(18, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(x_batch[i].reshape(48, 48), cmap='gray')
        label_index = np.argmax(y_batch[i])
        plt.title(f"Label: {class_names[label_index]}")
        plt.axis('off')
    plt.show()

# Get class names
class_names = list(train_generator.class_indices.keys())

# Visualize some samples
visualize_samples(train_generator)

# Count number of images in each class in training data
class_counts = dict.fromkeys(class_names, 0)

# Iterate over the batches of images
for i in range(len(train_generator)):
    _, labels = next(train_generator)
    for label in labels:
        class_index = np.argmax(label)
        class_name = class_names[class_index]
        class_counts[class_name] += 1

# Display class counts
for class_name, count in class_counts.items():
    print(f"{class_name}: {count} images")

# Alternatively, visualize the counts using a bar chart
plt.figure(figsize=(10, 6))
plt.bar(class_counts.keys(), class_counts.values(), color='violet')
plt.xlabel('Emotion Class')
plt.ylabel('Number of Images')
plt.title('Number of Images per Class')
plt.xticks(rotation=45)
plt.show()
