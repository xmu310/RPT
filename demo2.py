# pip install scikit-image
import sys
import cv2
import numpy as np
from skimage import io,color,transform,img_as_ubyte
from matplotlib import pyplot as plt
from runReal import runReal
I=io.imread(sys.argv[1])/255
if I.ndim==3:
	I=color.rgb2gray(I)
I=transform.resize(I,(65,65))

E=runReal(I)

#low pass filter
pupu_m, pupu_n = I.shape
pupu_low_I= np.zeros_like(I)
for i in range(1, pupu_m - 1):
    for j in range(1, pupu_n - 1):
        # Apply low pass filter (mean filter)
        pupu_low_I[i, j] = (I[i-1, j-1] + I[i-1, j] + I[i-1, j+1] +I[i, j-1]+ I[i, j]+ I[i, j+1] + I[i+1, j-1] + I[i+1, j] + I[i+1, j+1]) / 9
# pupu_low_I 應為low pass filter

# sobel
S = cv2.Sobel(pupu_low_I, -1, 1, 1, 1, 7)  

# canny
C = img_as_ubyte(pupu_low_I)
C = cv2.Canny(C, 100, 100)  

m=np.max(E)
E=E/(m+(m==0))
m=np.max(S)
S=S/(m+(m==0))
m=np.max(C)
C=C/(m+(m==0))

O=[I,E,S,C]
title=['input','output','sobel after\nlow pass','canny after\nlow pass']
figure,ax=plt.subplots(1,4)
for i in range(4):
	ax[i].set_title(title[i])
	ax[i].imshow(O[i],cmap='gray',vmin=0,vmax=1)
	ax[i].axis('off')
plt.show()

