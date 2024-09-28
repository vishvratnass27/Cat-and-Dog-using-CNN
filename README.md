Here’s a README.md file that explains the project, setup, and usage, along with placeholders for the Google Drive link for your dataset.

# CNN Cat and Dog Classifier using Keras

This project implements a Convolutional Neural Network (CNN) model using Keras to classify images of cats and dogs. 

The model is trained on the Cats and Dogs dataset and saved as an .h5 file, which can later be used for predictions.

# Dataset
The dataset used for training the model consists of images of cats and dogs, organized into folders for each class. 

You can download the dataset from the following link

# Folder Structure
# The dataset should be structured as follows:
cnn_dataset/
├── cat/       # Folder with cat images

├── dog/       # Folder with dog images

├── single_prediction/

│   └── cat_or_dog_1.jpg  # Image used for testing single predictions

# Model Architecture
The Convolutional Neural Network (CNN) has the following architecture:

Convolutional Layer: 32 filters, kernel size (3,3), ReLU activation
MaxPooling Layer: Pool size (2,2)
Second Convolutional Layer: 32 filters, kernel size (3,3), ReLU activation
MaxPooling Layer: Pool size (2,2)
Flatten Layer: Converts pooled feature maps into a vector
Dense Layer: 128 units, ReLU activation
Output Layer: 1 unit, Sigmoid activation (for binary classification)

# Code Overview

# Model Creation and Compilation

from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

# Initialize the model
model = Sequential()

# Add convolutional and pooling layers
model.add(Convolution2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten and add Dense layers
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data Preprocessing and Augmentation

from keras_preprocessing.image import ImageDataGenerator

# Data augmentation for training and testing
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training and test sets
training_set = train_datagen.flow_from_directory('cnn_dataset', target_size=(64, 64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory('cnn_dataset', target_size=(64, 64), batch_size=32, class_mode='binary')

# Training the Model

# Train the model for 2 epochs
model.fit(training_set, steps_per_epoch=8000, epochs=2, validation_data=test_set, validation_steps=800)

# Save the trained model
model.save('my.h5')

# Making Predictions

from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load the trained model
m = load_model('my.h5')

# Load and preprocess the test image
test_image = image.load_img('cnn_dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Make the prediction
result = m.predict(test_image)
if result[0][0] == 1.0:
    print('cat')
else:
    print('dog')

# How to Run
# 1. Install Dependencies
Make sure you have Python installed. Then, install the required dependencies using pip:

pip install tensorflow keras keras_preprocessing numpy

# 2. Download Dataset
Download the dataset from this link and ensure it's structured correctly (see the Folder Structure section).

# 3. Train the Model
Run the following script to train the model and save it as my.h5:

python train_model.py

# 4. Test with a Single Image
After training, you can test the model with a single image by running:

python test_single_image.py

Make sure the image is placed in the correct folder (cnn_dataset/single_prediction/).


# Notes

Adjust Dataset Paths: Ensure the paths to the dataset and images are correct relative to your project directory.

Epochs and Steps: You may need to adjust steps_per_epoch and validation_steps based on the size of your dataset.
