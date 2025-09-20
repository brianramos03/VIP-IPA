import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

#creates gaussian kernel to size img with specified sigma
def gaussian_kernel2(sigma, img):
    x, y = img.shape
    sigma = 30
    centerx, centery = x//2, y//2
    h = np.zeros((x, y), dtype=np.float32)
    for i in range(x):
        for j in range(y):
            #generates centered gaussian kernel according to equation
            h[i, j] = np.exp(-((i-centerx)**2 + (j-centery)**2) / (2*sigma**2))
    return h

#creates gaussian kernel to specified size and sigma padded to image size
def gaussian_kernel(sigma, size, img):
    x, y = img.shape
    kernel = np.zeros((x, y), dtype=np.float32)
    centerx, centery = y//2
    for i in range(size):
        for j in range(size):
            kernel[i,j] = np.exp(-((i-centerx)**2 + (j-centery)**2) / (2*sigma**2))
    #kernel /= np.sum(kernel)
    print(kernel)
    return kernel



#reads image f
f = cv2.imread("./unblurry2.jpg", cv2.IMREAD_GRAYSCALE)
#creates gaussian kernel

#h = gaussian_kernel(30, 7, f)
h = gaussian_kernel2(30, f)

#converts image f in spatial to fourier
F = fft2(f)
F = fftshift(F)

#displays fourier of image and kernel for testing
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("fourier of image")
plt.imshow(np.log(np.abs(F) + 1), cmap='gray')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("fourier of kernel")
plt.imshow(np.log(np.abs(h) + 1), cmap='gray')
plt.colorbar()

plt.show()

#multiplies image F by gaussian filter h in the fourier domain
G = F * h

#converts fourier back into spatial resulting in degraded image g
G2 = ifftshift(G)
g = ifft2(G2)
g = np.abs(g)

result_image = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#applies weiner formula to fourier of image
#0.01 is arbritrary value we choose close to zero because we do not know the noise function of the degraded image

F_hat = (G * np.conj(h) / (np.abs(h)**2 + 0.01))

#F_hat = (1/h) * ((G  * np.abs(h)**2) /  (np.abs(h)**2 + 0.01))

#converts fourier back into spatial
f_hat = ifft2(F_hat)
f_hat = np.abs(f_hat)

plt.figure(figsize = (12,5))
plt.subplot(131)
plt.title("original image")
plt.imshow(f, cmap = 'gray', vmin=0, vmax=255)
plt.subplot(132)
plt.title("degraded image")
plt.imshow(g, cmap = 'gray', vmin=0, vmax=255)
plt.subplot(133)
plt.title("restored image")
plt.imshow(f_hat, cmap = 'gray', vmin=0, vmax=255)
plt.show()

#reads image f
f = cv2.imread("./blurry1.jpg", cv2.IMREAD_GRAYSCALE)
F = fft2(f)
F = fftshift(F)

#deconvolutes blurry image with degradation function from other image for testing
#plan to use this
F_hat = (G * np.conj(h) / (np.abs(h)**2 + 0.01))
f_hat = ifft2(F_hat)
f_hat = np.abs(f_hat)

plt.figure(figsize = (12,5))
plt.subplot(121)
plt.title("blurred image")
plt.imshow(f_hat, cmap = 'gray', vmin=0, vmax=255)
plt.subplot(122)
plt.title("restored image")
plt.imshow(f_hat, cmap = 'gray', vmin=0, vmax=255)
#plt.show()