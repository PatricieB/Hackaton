# Importy knihoven
import os
# from keras.datasets import mnist
import numpy as np
import tensorflow as tf
from skimage.filters import threshold_otsu
from skimage import exposure
import cv2
to_create = {
    'root': '/Data',
    'train_dir': 'Data/train_set',
    'test_dir': 'Data/test_set/',
    'true_train_dir': 'Data/train_set/solar',
    'false_train_dir': 'Data/train_set/non_solar',
    'true_test_dir': 'Data/test_set/solar',
    'false_test_dir': 'Data/test_set/non_solar',

}




def data_load(root_path, scale=(256,256)):
  categories =  os.listdir(root_path)
  x = []
  y =[]
  for i, cat in enumerate(categories):
    img_path = os.path.join(root_path, cat)
    images = os.listdir(img_path)
    for image in images:
      img = cv2.imread(os.path.join(img_path, image), 0)
      img = cv2.resize(img, scale)
      x.append(img)
      y.append(i)
  return np.array(x), np.array(y)
x_train, y_train = data_load(to_create.get('train_dir'))
x_test, y_test = data_load(to_create.get('test_dir'))
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)


model =  tf.keras.Sequential([

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

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=2,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
