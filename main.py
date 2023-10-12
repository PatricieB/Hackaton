# Importy knihoven
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
# from keras.datasets import mnist


data = iio.imread('train_set/solar/AABCN78E1YHCWW.png')


# Rozměry načtených obrázků
print(f"Shape of test images: {data.shape}")
print(f"Shape of an individual image: {data[0].shape}")

# Zobrazení prvního příkladu ve formě obrázku
plt.imshow(data, cmap='gray')
#plt.title(f"Label: {test_labels[0]}")
plt.axis("off")
plt.show()

# Tisk prvního příkladu ve formě matice
print(f"Matrix representation of the above image:\n")
for row in data[0]:
    print(' '.join([f'{num:3}' for num in row]))