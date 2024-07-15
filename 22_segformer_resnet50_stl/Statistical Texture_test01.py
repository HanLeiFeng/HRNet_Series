import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('img/library.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Compute texture gradient
texture_grad = cv2.absdiff(gray, blur)

# Display results
plt.subplot(131), plt.imshow(image, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(blur, cmap = 'gray')
plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(texture_grad, cmap = 'gray')
plt.title('Texture Gradient'), plt.xticks([]), plt.yticks([])
plt.show()