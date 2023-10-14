# Importy knihoven
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import cv2
import os
# from keras.datasets import mnist
import tensorflow as tf
from skimage.filters import threshold_otsu
from skimage import exposure
to_create = {
    'root': '/Data',
    'train_dir': 'Data/train_set',
    'test_dir': 'Data/test_set/',
    'true_train_dir': 'Data/train_set/solar',
    'false_train_dir': 'Data/train_set/non_solar',
    'true_test_dir': 'Data/test_set/solar',
    'false_test_dir': 'Data/test_set/non_solar',

}

from tensorflow.keras.preprocessing.image import ImageDataGenerator

kwargs = dict(
    featurewise_center=False,
    featurewise_std_normalization=False,
    # rotation_range=90,
    rescale=1. / 255,
    # zca_whitening=True,
    # horizontal_flip=True,
    # vertical_flip=True,
    # preprocessing_function=preprocessing
)

train_datagen = ImageDataGenerator(**kwargs)

train_tfdata = train_datagen.flow_from_directory(directory=to_create.get('train_dir'),
                                                 seed=24,
                                                 color_mode='grayscale',
                                                 batch_size=6, class_mode='binary'
                                                 )
test_datagen = ImageDataGenerator(rescale=1. / 255,
                                  )
test_tfdata = test_datagen.flow_from_directory(directory=to_create.get('test_dir'),
                                               seed=24,color_mode='grayscale',
                                                                        batch_size=2,class_mode='binary'
                                                                        )
history = train_datagen.fit(train_tfdata)
print(history)
