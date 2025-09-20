import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

#creates gaussian kernel to specified size and sigma padded to image size
def gaussian_kernel(sigma, img):
    x, y = img.shape
    sigma = 30
    centerx, centery = x//2, y//2
    h = np.zeros((x, y), dtype=np.float32)
    for i in range(x):
        for j in range(y):
            #generates centered gaussian kernel according to equation
            h[i, j] = np.exp(-((i-centerx)**2 + (j-centery)**2) / (2*sigma**2))
    return h

def richardson_lucy(image, psf, iterations):
    f_hat = image
    H = fft2(psf)

    for i in range(iterations):
        F_hat = fft2(f_hat)
        #X = F_hat * h
        X = F_hat * H
        x = ifft2(X)

        blur = image / (x + 1e-6)
        #correction_fft = blur * np.conj(h)
        correction_fft = blur * np.conj(H)
        correction = ifft2(correction_fft)

        f_hat *= np.real(correction)
    return np.real(f_hat)

f_hat = cv2.imread("./unblurry1.jpg", cv2.IMREAD_GRAYSCALE)
f_hat = f_hat.astype(np.float64) / 255.0

plt.figure(figsize=(5, 5))
plt.title("original image")
plt.imshow(f_hat, cmap='gray')
plt.show()

h = (gaussian_kernel(30, f_hat))

unblurred = richardson_lucy(f_hat, h, 2)

unblurred = np.clip(unblurred * 255, 0, 255).astype(np.uint8)

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.title("original image")
plt.imshow(f_hat, cmap='gray')
plt.subplot(122)
plt.title("richardson-lucy")
plt.imshow(unblurred, cmap='gray')
plt.show()
