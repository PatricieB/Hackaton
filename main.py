# Importy knihoven
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import cv2
# from keras.datasets import mnist

from skimage.filters import threshold_otsu
from skimage import exposure



originalImage = cv2.imread('train_set/solar/AABCN78E1YHCWW.png')
grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

# Histogram equalization na prvním příkladu
#equalized_image = exposure.equalize_hist(data)

ada_thresh_gaussian = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 4)

# Zobrazení před a po

plt.subplot(1, 2, 2)
plt.imshow(ada_thresh_gaussian, cmap='gray')
plt.title("Equalized Image")
plt.axis("off")

plt.show()






