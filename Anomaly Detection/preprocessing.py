import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("Shiny Silver Hex Bolt.png")

if image is None:
    raise ValueError("Image not found!")

resized = cv2.resize(image, (224, 224))

gray = cv2.cvtColor(
    resized, cv2.COLOR_BGR2GRAY
)

blurred = cv2.GaussianBlur(
    gray, (5, 5), 0
)

edges = cv2.Canny(
    blurred, 100, 200
)

normalized = resized.astype(
    np.float32
) / 255.0

print("Original shape:", image.shape)
print("Resized shape:", resized.shape)
print("Grayscale shape:", gray.shape)

print("Normalized min:", normalized.min())
print("Normalized max:", normalized.max())

images = [
    cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
    gray,
    blurred,
    edges
]

titles = [
    "Original",
    "Grayscale",
    "Gaussian Blur",
    "Edge Detection"
]

plt.figure(figsize=(12, 7))

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