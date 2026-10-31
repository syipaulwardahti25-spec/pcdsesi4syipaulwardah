import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('kelinci_noise.jpg')
if img is None:
    raise FileNotFoundError("kelinci_noise.jpg tidak ditemukan")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Smoothing dengan Mean Filter
mean = cv2.blur(rgb, (5, 5))

# Smoothing dengan Median Filter
median = cv2.medianBlur(rgb, 5)

# Smoothing dengan Gaussian Filter
gaussian = cv2.GaussianBlur(rgb, (5, 5), 0)

fig, ax = plt.subplots(1, 4, figsize=(16, 4))

for a, im, title in zip(
    ax,
    [rgb, mean, median, gaussian],
    ['Noise', 'Mean Filter', 'Median Filter', 'Gaussian Filter']
):
    a.imshow(im)
    a.set_title(title)
    a.axis('off')

plt.tight_layout()
plt.show()