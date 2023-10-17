# Importy knihoven
import os
# from keras.datasets import mnist
import numpy as np
import tensorflow as tf
from keras import backend as k
from skimage.filters import threshold_otsu
from skimage import exposure
import cv2
from sklearn.metrics import classification_report

# Paths to directories
to_create = {
    'root': '/Data',
    'train_dir': 'Data/train_set',
    'test_dir': 'Data/test_set/',
    'true_train_dir': 'Data/train_set/solar',
    'false_train_dir': 'Data/train_set/non_solar',
    'true_test_dir': 'Data/test_set/solar',
    'false_test_dir': 'Data/test_set/non_solar',

}


# Function to load data to np.array
def data_load(root_path, scale=(256, 256)):
    categories = os.listdir(root_path)
    x = []
    y = []
    for i, cat in enumerate(categories):
        img_path = os.path.join(root_path, cat)
        images = os.listdir(img_path)
        for image in images:
            img = cv2.imread(os.path.join(img_path, image), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, scale)

            x.append(img)
            y.append(i)
    return np.array(x), np.array(y)

# Loading data to variables
x_train, y_train = data_load(to_create.get('train_dir'))
x_test, y_test = data_load(to_create.get('test_dir'))

# Adding dimension
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

# Creating model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, 3, strides=(1, 1),
                           activation='relu', padding='same',
                           input_shape=[256, 256, 1]),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(128, 3, strides=(1, 1),
                           activation='relu', padding='same',
                           ),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])

# Compiling model
model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Training model
model.fit(x_train, y_train,
          # batch_size=128,
          epochs=2,
          verbose=1,
          validation_data=(x_test, y_test))
# Evaluating
score = model.evaluate(x_test, y_test, verbose=1)

# F1 Score
f1score = 2 * (score[1] * score[2]) / (score[1] + score[2])
print(score)
print(f1score)
