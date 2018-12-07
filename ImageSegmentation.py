# importing import Libraries
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Reading Image Using openCV
img = cv2.imread('elephant.jpg')
cvt_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
orig_shape = img.shape
""" Uncomment to see the Shape of orginal Image 
	print(img.shape) """

# Plotting the image Using matplotlib
plt.imshow(cvt_img)
plt.show()

# Creating dataset
all_pixels = cvt_img.reshape((-1,3))
""" Uncomment to See the shape of the dataset 
	print(all_pixels.shape) """

# Using Sklearn Kmeans Algorithm

# Change the Value According to you
dominant_colors = 7

# Instantiating Object of Kmeans class
km = KMeans(n_clusters=dominant_colors)

# Training the model
km.fit(all_pixels)

""" Uncomment To check the labels in the model
	km.labels_ """

# To get the coordinates of the centers in type-int
centers = km.cluster_centers_
centers = np.array(centers,dtype='uint8')
""" Uncomment to see the centers predicted 
	print(centers) """

# Printing the Color Swatches of dominating Colors
i = 1

# To store the K most dominant Colors
colors = []

# Looping over each color in centers
for each_col in centers:
    
    # To Create the Subplots using matplotlib
    plt.subplot(1,dominant_colors,i)
    plt.axis('off')
    
    colors.append(each_col)
    
    # Colors Swatch
    Swatch = np.zeros((100,100,3),dtype='uint8')

    # Filling the Swatch
    Swatch[:,:,:] = each_col
    
    # plotting the swatches using Matplotlib
    plt.imshow(Swatch)

    i+= 1

plt.show()

""" Uncomment to see the Colors array created
	print(colors) """

# Reconstructin the same image with only most dominating K colors

# Creating New Image dimensions 
new_image = np.zeros((330*500,3),dtype="uint8")
""" Uncommnet to the Shape of the newly Created Image 
	print(new_image.shape) """

# Rebuilding the whole image using the most K dominant colors
for ix in range(new_image.shape[0]):
    new_image[ix] = colors[km.labels_[ix]]

# Reshaping the new_image to print the image formed
new_image = new_image.reshape((orig_shape))

# Plotting the image using matplotlib
plt.imshow(new_image)
plt.show()

# Thanks Guys! Happy coding ... :p