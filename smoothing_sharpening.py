import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


# Membaca citra noise
img = iio.imread('kelinci_noise.jpg')

# Jika gambar RGBA, ambil RGB saja
if img.shape[-1] == 4:
    img = img[:, :, :3]

# Pastikan tipe data uint8
img = img.astype(np.uint8)


# ==============================
# SMOOTHING
# ==============================

# Mean Filter
mean = ndimage.uniform_filter(
    img,
    size=(5, 5, 1)
)

# Median Filter
median = ndimage.median_filter(
    img,
    size=(5, 5, 1)
)

# Gaussian Filter
gaussian = ndimage.gaussian_filter(
    img,
    sigma=(1, 1, 0)
)


# ==============================
# SHARPENING
# ==============================

kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])


# Sharpening Mean
sharp_mean = ndimage.convolve(
    mean.astype(float),
    kernel[:, :, np.newaxis],
    mode='reflect'
)

# Sharpening Median
sharp_median = ndimage.convolve(
    median.astype(float),
    kernel[:, :, np.newaxis],
    mode='reflect'
)

# Sharpening Gaussian
sharp_gaussian = ndimage.convolve(
    gaussian.astype(float),
    kernel[:, :, np.newaxis],
    mode='reflect'
)


# Membatasi nilai piksel 0-255
sharp_mean = np.clip(sharp_mean, 0, 255).astype(np.uint8)
sharp_median = np.clip(sharp_median, 0, 255).astype(np.uint8)
sharp_gaussian = np.clip(sharp_gaussian, 0, 255).astype(np.uint8)


# ==============================
# MENAMPILKAN HASIL
# ==============================

fig, ax = plt.subplots(2, 4, figsize=(16, 8))


# Baris pertama
images1 = [
    img,
    mean,
    median,
    gaussian
]

titles1 = [
    'Noise',
    'Mean Filter',
    'Median Filter',
    'Gaussian Filter'
]

for a, image, title in zip(ax[0], images1, titles1):
    a.imshow(image)
    a.set_title(title)
    a.axis('off')


# Baris kedua
images2 = [
    sharp_mean,
    sharp_median,
    sharp_gaussian
]

titles2 = [
    'Sharpening Mean',
    'Sharpening Median',
    'Sharpening Gaussian'
]

for a, image, title in zip(ax[1], images2, titles2):
    a.imshow(image)
    a.set_title(title)
    a.axis('off')


# Kotak terakhir dikosongkan
ax[1, 3].axis('off')


plt.tight_layout()
plt.show()