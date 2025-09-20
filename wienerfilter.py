import cv2
import matplotlib.pyplot as plt
import numpy as np

from numpy.fft import fft2, ifft2, fftshift, ifftshift

def grayscale(img):
    gray_image = np.zeros(np.shape(img)).astype(np.uint8)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            pixel = img[y][x]
            R,G,B = pixel
            gray_val = 0.2126*R+0.7152*G+0.0722*B 
            gray_image[y][x] = [gray_val, gray_val, gray_val] 
    return gray_image

def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))  # Create an empty matrix for the kernel
    center = size // 2  # Calculate the center index
    
    # Constant for the Gaussian formula (normalization factor)
    normalizing_constant = 1 / (2 * np.pi * sigma ** 2)
    
    # Populate the kernel matrix using the Gaussian formula
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center  # Calculate distance from center
            kernel[i, j] = normalizing_constant * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Normalize the kernel so that the sum of all elements is 1
    kernel /= np.sum(kernel)
    
    return kernel

def wiener_filter(blurred_image, psf, K):
    # Convert the blurred image and PSF to frequency domain
    G = fft2(blurred_image)
    
    H = fft2(psf, s=blurred_image.shape)

    print(psf)
    
    # Apply Wiener filter formula
    H_conj = np.conj(H)
    Wiener_filter = H_conj / (H_conj * H + K)
    
    # Apply the filter in frequency domain and inverse FFT to get the restored image
    F_hat = Wiener_filter * G
    restored_image = np.abs(ifft2(F_hat))
    
    return restored_image

image = cv2.imread("./unblurry1.jpg", cv2.IMREAD_GRAYSCALE)
original_image = image

psf = gaussian_kernel(7, 3)
K = 1

restored_image = wiener_filter(image, psf, K)

#print("image shape:", image.shape)


plt.figure(figsize=(10, 8))
plt.subplot(1,2,1)
plt.title("original")
plt.imshow(original_image, cmap="gray")
plt.subplot(1,2,2)
plt.title("weiner filter")
plt.imshow(image, cmap="gray")
plt.show()

