import cv2
from PIL import Image
import numpy as np
from numpy import linalg as LA


def dot(a, b):
	total =0
	for i in range(a.size):
		total = total + int(a[i])*int(b[i])
	return total

def compute_Similarity(input):
	my_image = cv2.imread(input, 0)
	my_image = np.array(my_image)
	shortest = my_image.size
	my_image = my_image.reshape(1, my_image.size)

#his_image = cv2.imread('he.jpg', 0)
#his_image = np.array(his_image)

	my_image2 = cv2.imread('me.png', 0)
	my_image2 = np.array(my_image2)
	my_image2 = my_image2.reshape(1, my_image2.size)

	shortest = min(shortest, my_image2.size)
	my_image = my_image.flatten()[: shortest]
	my_image_norm = LA.norm(my_image)
	my_image2 = my_image2.flatten()[: shortest]
	my_image2_norm = LA.norm(my_image2)

	res = dot(my_image, my_image2)/(my_image_norm* my_image2_norm)
	#print (res)
	return res
# maybe need some slice:  img.reshape(1,**), kid_image.flatten() to 1D.  kid_image[0:100]. np.dot(...)
