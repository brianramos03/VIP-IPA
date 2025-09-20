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

def richardson_lucy_fourier(blurred_image, kernel, iterations):
    # Ensure image and kernel are in float64 format for numerical stability
    blurred_image = blurred_image.astype(np.float64)
    estimate = np.full_like(blurred_image, 0.5)  # Initial estimate

    # Fourier transform of the kernel
    kernel_fft = np.fft.fft2(kernel)
    
    # Fourier transform of the blurred image
    blurred_image_fft = np.fft.fft2(blurred_image)
    
    # Iterate the Richardson-Lucy algorithm
    for i in range(iterations):
        # Step 1: Convolve the current estimate with the kernel (in Fourier domain)
        estimate_fft = np.fft.fft2(estimate)
        convolved_estimate_fft = kernel_fft * estimate_fft
        convolved_estimate = np.fft.ifft2(convolved_estimate_fft)
        
        # Step 2: Compute the ratio of the blurred image to the convolved estimate
        relative_blur = blurred_image / (convolved_estimate + 1e-6)  # Avoid division by zero
        
        # Step 3: Convolve the result with the flipped kernel (using Fourier domain)
        relative_blur_fft = np.fft.fft2(relative_blur)
        kernel_flipped_fft = np.conj(kernel_fft)  # Conjugate of the Fourier transform of the kernel
        correction_fft = relative_blur_fft * kernel_flipped_fft
        correction = np.fft.ifft2(correction_fft)
        
        # Step 4: Update the estimate
        estimate *= np.real(correction)
    
    return np.real(estimate)

f_hat = cv2.imread("./unblurry1.jpg", cv2.IMREAD_GRAYSCALE)
kernel = gaussian_kernel(20, f_hat)

iterations = 2
deblurred_image = richardson_lucy_fourier(f_hat, kernel, iterations)

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.title("original image")
plt.imshow(f_hat, cmap='gray')
plt.subplot(122)
plt.title("richardson-lucy")
plt.imshow(deblurred_image, cmap='gray')
plt.show()
