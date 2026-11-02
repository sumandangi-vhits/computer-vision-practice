import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("Shiny Silver Hex Bolt.png")

if image is None:
    raise ValueError("Image not found!")

image = cv2.resize(image, (224, 224))
gray = cv2.cvtColor(
    image, cv2.COLOR_BGR2GRAY
)

edges = cv2.Canny(gray, 100, 200)

sobel = cv2.Sobel(
    gray, cv2.CV_64F, 1, 1, ksize=3
)

_, threshold = cv2.threshold(
    gray, 127, 255, cv2.THRESH_BINARY
)

contours, _ = cv2.findContours(
    threshold,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

mean = np.mean(gray)
std = np.std(gray)

print("Mean intensity:", mean)
print("Standard deviation:", std)
print("Number of contours:", len(contours))

images = [
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    gray,
    edges,
    sobel
]

titles = [
    "Original Image",
    "Grayscale",
    "Edge Features",
    "Sobel Features"
]

plt.figure(figsize=(10, 8))

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(
        images[i],
        cmap="gray" if i > 0 else None
    )
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()