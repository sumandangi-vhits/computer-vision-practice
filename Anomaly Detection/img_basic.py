import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("Shiny Silver Hex Bolt.png")

if image is None:
    raise ValueError("Image not found!")

print("Image shape:", image.shape)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

resized = cv2.resize(image_rgb, (224, 224))

gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(resized)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")

plt.show()