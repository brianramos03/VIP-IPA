import cv2      #importing libraries
import math
import matplotlib.pyplot as plt
import numpy as np

def mod(n, m):
        return ((n%m)+m)%m

def pass_filter(img, filter):
    new_image = np.zeros(np.shape(img), dtype=np.int64) # create new black image

    size_offset = int(len(filter)/2)
                                                                    # Loop for every...
    for y in range(img.shape[0]):        # row
        for x in range(img.shape[1]):    # pixel in row
            for c in range(img.shape[2]):                           # color in pixel

                y_min = max(0, y - size_offset)
                y_max = min(img.shape[0], y + size_offset + 1)
                x_min = max(0, x - size_offset)
                x_max = min(img.shape[1], x + size_offset + 1)

                channel_subset = img[y_min:y_max, x_min:x_max, c]       #apply filter, handle edges by mirroring to prevent black edges

                if y - size_offset < 0:         
                    channel_subset = np.vstack((np.flipud(channel_subset[:size_offset, :]), channel_subset))
                if y + size_offset >= img.shape[0]:
                    channel_subset = np.vstack((channel_subset, np.flipud(channel_subset[-size_offset:, :])))

                if x - size_offset < 0:
                    channel_subset = np.hstack((np.fliplr(channel_subset[:, :size_offset]), channel_subset))
                if x + size_offset >= img.shape[1]:
                    channel_subset = np.hstack((channel_subset, np.fliplr(channel_subset[:, -size_offset:])))

                new_image[y][x][c] = int(np.sum(np.multiply(filter, channel_subset)))
    return new_image


def normalize(new_image):				#this is only for displaying the image
    minval = np.min(new_image)                      
    maxval = np.max(new_image - minval)
    if (maxval != 0):
        new_image = ((new_image - minval) / maxval)
    return new_image

def normal(new_image):				#this is only for displaying the image, divides by 
    new_image = new_image / 256
    return new_image

img = cv2.imread("./blurry1.jpg") #upload images, then make color adjustments
img = np.flip(img, 2)

gaussian = np.array([[0, 1, 0],\
                    [ 1, 1, 1],\
                    [ 0, 1, 0]])     #laplacian

blurred = pass_filter(img, gaussian)
sharp = img + (img - (blurred/5))*5
sharp = normal(sharp)

plt.figure(figsize=(16, 4))     #make figure with 3 spots
plt.subplot(1,4,1)
plt.title("original")
plt.imshow(img)
plt.subplot(1,4,2)
plt.title("sharpened")
plt.imshow(sharp)
plt.show()