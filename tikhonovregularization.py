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

#performs richardson lucy algorithm for psf
#with degraded image f, current image f_hat, current PSF h_hat for k iterations
def richardson_lucy_psf(f, f_hat, h_hat, k):
    for i in range(k):
        F_hat = fft2(f_hat)
        F_hat = fftshift(F_hat)
        #calculates blur ratio X
        X = F_hat * h_hat
        X = ifftshift(X)
        x = ifft2(X)
        x = np.abs(x)
        #
        y = f / x
        Y = fft2(y)
        Y = fftshift(Y)
        #
        Z = np.conj(h_hat) * Y
        Z = ifftshift(Z)
        z = ifft2(Z)
        z = np.abs(z)
        #
        h_hat *= z
    return h_hat

def richardson_lucy_img(f, f_hat, h_hat, k):
    for i in range(k):
        F_hat = fft2(f_hat)
        F_hat = fftshift(F_hat)
        #calculates blur ratio X
        X = F_hat * h_hat
        X = ifftshift(X)
        x = ifft2(X)
        x = np.abs(x)
        #
        y = f / x
        Y = fft2(y)
        Y = fftshift(Y)
        #
        Z = np.conj(h_hat) * Y
        Z = ifftshift(Z)
        z = ifft2(Z)
        z = np.abs(z)
        #
        f_hat *= z
    return f_hat

#performs richardson lucy algorithm from degraded image f with PSF h for inner loop j iterations and outer loop k iterations
def richardson_lucy(f, h, j, k):
    #initial estimate f_hat
    f_hat = f.astype(np.float64)
    #initial estimate h_hat
    h_hat = h
    for i in range (k):
        h_hat = richardson_lucy_psf(f, f_hat, h_hat, j)
        f_hat = richardson_lucy_img(f, f_hat, h_hat, j)
    return np.real(f_hat)

#performs richardson lucy algorithm from degraded image f with PSF h for inner loop j iterations until tikhonov error exceeds previous error
#performs a minimum of min iterations
def richardson_lucy_tikhonov(f, h, j, min):
    #initializes alpha
    #alpha is a coefficient that determines how much the error is dependent on similarity to signal 
    alpha = 0.01

    #initializes error
    error = 0.5 * np.sum((f - h * f) ** 2)
    print("initial error:", error)

    #initial estimate of image f_hat
    f_hat = f.astype(np.float64)
    #initial estimate of psf h_hat
    h_hat = h


    i = 0
    while(True):
        h_hat = richardson_lucy_psf(f, f_hat, h_hat, j)
        f_hat = richardson_lucy_img(f, f_hat, h_hat, j)
        i = i + 1
        next_error = (0.5 * np.sum((f - h * f_hat) ** 2)) + (alpha * (np.sum(((f_hat - f) / (f + 1e-10) ** 2))))
        print("iteration ", i, " error: ", next_error)
        #returns current estimate for image if error is greater than previous error and minimum iterations have been reached
        if(error < next_error):
            print("current error greater than previous error")
            if(i >= min):
                print("minimum ", min, " iterations reached")
                return np.real(f_hat), i
        if(i >= 200):
            print("maximum 200 iterations reached")
            return np.real(f_hat), i
        error = next_error

#f = cv2.imread("./sharp/269_NIKON-D3400-18-55MM_S.JPG", cv2.IMREAD_GRAYSCALE)
#f_hat = cv2.imread("./defocused_blurred/269_NIKON-D3400-18-55MM_F.JPG", cv2.IMREAD_GRAYSCALE)

f = cv2.imread("./unblurry1.jpg", cv2.IMREAD_GRAYSCALE)
f_hat = cv2.imread("./blurry1.jpg", cv2.IMREAD_GRAYSCALE)

f_hat_t = f_hat

#f = cv2.imread("./unblurry1.jpg", cv2.IMREAD_GRAYSCALE)
#f_hat = cv2.imread("./blurry1.jpg", cv2.IMREAD_GRAYSCALE)

h = (gaussian_kernel(30, f_hat))
h_t = (gaussian_kernel(30, f_hat_t))

inner = 2
outer = 30

unblurred_tikhonov, iterations = richardson_lucy_tikhonov(f_hat_t, h_t, inner, 10)
unblurred = richardson_lucy(f_hat, h, inner, outer)

unblurrednormalized_tikhonov = cv2.normalize(unblurred_tikhonov, None, 0, 255, cv2.NORM_MINMAX)
unblurrednormalized = cv2.normalize(unblurred, None, 0, 255, cv2.NORM_MINMAX)


plt.figure(figsize=(12, 5))
plt.subplot(131)
plt.title("degraded image")
plt.imshow(f_hat, cmap='gray')
plt.axis("off")
plt.subplot(132)
plt.title(f"RL stopped after {iterations} iterations")
plt.imshow(unblurrednormalized_tikhonov, cmap='gray')
plt.axis("off")
plt.subplot(133)
plt.title(f"RL {outer} iterations")
plt.imshow(unblurrednormalized, cmap='gray')
plt.axis("off")
plt.show()

#plt.figure(figsize=(12, 5))
#plt.subplot(121)
#plt.title("degraded image")
#plt.imshow(f_hat, cmap='gray')
#plt.axis("off")
#plt.subplot(122)
#plt.title(f"RL stopped after {iterations} iterations")
#plt.imshow(unblurrednormalized_tikhonov, cmap='gray')
#plt.axis("off")
#plt.show()
