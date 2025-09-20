import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math

imageSharp = cv.imread("./hopper.jpeg",  cv.IMREAD_GRAYSCALE)
imageShould = cv.imread("./hopper.jpeg",  cv.IMREAD_GRAYSCALE)

def fourierTransform(image):

    #
    # Can only do grayscale pics
    # 

    new_image = np.zeros(np.shape(image), dtype=np.complex64) # create black image
  
    # Size Var
    xSize = image.shape[1]
    ySize = image.shape[0]

    
    # 2D-DFT
    # Very Slow (100x100px (Grayscale) = 3mins)
    # Work on optimitization next
    
    for y in range(ySize):
        # Checks progress of function
        print(f"{y}/{ySize}")
        for x in range(xSize):
            sum = 0
            for x2 in range(int(xSize)):
                
                for y2 in range(int(ySize)):

                    co = ((x*x2/xSize) + (y2*y)/(ySize))

                    power: complex = math.e**((-2j*math.pi*co))
                    # Multiplys value of pixel to coefficient
                    sum += image[y2,x2]*power

            new_image[y,x] = sum

    image = new_image
    
    # Shifts image, making top left as center
    image = np.fft.fftshift(image)
    
    return image


imageSharp = fourierTransform(imageSharp)

imageSharp = 20*np.log(np.abs(imageSharp))

#
# To compare 2D Dft function to OpenCV Dft functiono (fft2)
#
f2 = np.fft.fft2(imageShould)
fshift2 = np.fft.fftshift(f2)
magnitude_spectrum = 20*np.log(np.abs(fshift2))


print("done")
plt.figure(figsize=(10, 8))
plt.subplot(1,2,1)
plt.title("WIP Image")
plt.imshow(imageSharp, cmap="gray")
plt.subplot(1,2,2)
plt.title("Should be Image")
plt.imshow(magnitude_spectrum, cmap="gray")
plt.show()