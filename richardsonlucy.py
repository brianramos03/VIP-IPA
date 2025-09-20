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

#performs richardson_lucy algorithm from degraded image f with PSF h for k iterations
def richardson_lucy(f, h, k):
    #initial estimate f_hat
    f_hat = f.astype(np.float64)
    #Fourier transform of PSF h
    H = fft2(h)
    for i in range (k):
        #Fourier transform of initial estimate
        F_hat = fft2(f_hat)
        F_hat = fftshift(F_hat)
        #product X from convolution operation in fourier domain with PSF h and image f
        X = F_hat * h
        #X = F_hat * H
        #fourier transformation into spatial domain
        X = ifftshift(X)
        x = ifft2(X)
        x = np.abs(x)
        #blur ratio according to richardson lucy equation
        y = f / x
        Y = fft2(y)
        Y = fftshift(Y)
        Z = np.conj(h) * Y
        #Z = np.conj(H) * Y

        Z = ifftshift(Z)
        z = ifft2(Z)
        z = np.abs(z)

        f_hat *= np.real(z)

        

    return np.real(f_hat)
    

f_hat = cv2.imread("./blurry1.jpg", cv2.IMREAD_GRAYSCALE)
h = (gaussian_kernel(30, f_hat))

#displays kernel for testing
#plt.figure(figsize=(5, 5))
#plt.subplot(1,1,1)
#plt.title("kernel")
#plt.imshow(np.log(np.abs(h) + 1), cmap='gray')
#plt.axis("off")
#lt.show()

iterations = 10
unblurred = richardson_lucy(f_hat, h, iterations)
iterations2 = 50
unblurred2 = richardson_lucy(f_hat, h, iterations2)

plt.figure(figsize=(12, 5))
plt.subplot(131)
plt.title("original image")
plt.imshow(f_hat, cmap='gray')
plt.axis("off")
plt.subplot(132)
plt.title(f"richardson-lucy {iterations} iterations")
plt.imshow(unblurred, cmap='gray')
plt.axis("off")
plt.subplot(133)
plt.title(f"richardson-lucy {iterations2} iterations")
plt.imshow(unblurred2, cmap='gray')
plt.axis("off")
plt.show()

#plt.figure(figsize=(12, 5))
#plt.subplot(121)
#plt.title("original image")
#plt.imshow(f_hat, cmap='gray')
#plt.subplot(122)
#plt.title(f"richardson-lucy {iterations} iterations")
#plt.imshow(unblurred, cmap='gray')
#plt.show()






