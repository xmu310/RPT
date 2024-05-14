import sys
import cv2
import numpy as np
from skimage import io,color,transform,img_as_ubyte
from matplotlib import pyplot as plt
from runReal import runReal

def butterworth_low_pass_filter(image, cutoff, order):
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    gray_image = np.float32(gray_image)
    dft = cv2.dft(gray_image, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = gray_image.shape
    x = np.linspace(-0.5, 0.5, cols)
    y = np.linspace(-0.5, 0.5, rows)
    x, y = np.meshgrid(x, y)
    radius = np.sqrt((x - 0.5)**2 + (y - 0.5)**2) 
    filter = 1 / (1 + (radius / cutoff)**(2*order))
    filtered_dft = dft_shift * filter[:,:,np.newaxis]
    filtered_image_dft = np.fft.ifftshift(filtered_dft)
    filtered_image = cv2.idft(filtered_image_dft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    return filtered_image

I=io.imread(sys.argv[1])/255
if I.ndim==3:
	I=color.rgb2gray(I)
I=transform.resize(I,(65,65))

E=runReal(I)

S = cv2.Sobel(I, -1, 1, 1, 1, 7)  
C = img_as_ubyte(I)
C = cv2.Canny(C, 100, 100)  

pupu_m, pupu_n = I.shape
pupu_low_I= np.zeros_like(I)
for i in range(1, pupu_m - 1):
    for j in range(1, pupu_n - 1):
        pupu_low_I[i, j] = (I[i-1, j-1] + I[i-1, j] + I[i-1, j+1] +I[i, j-1]+ I[i, j]+ I[i, j+1] + I[i+1, j-1] + I[i+1, j] + I[i+1, j+1]) / 9

S1 = cv2.Sobel(pupu_low_I, -1, 1, 1, 1, 7)  
C1 = img_as_ubyte(pupu_low_I)
C1 = cv2.Canny(C1, 100, 100) 

I2 = butterworth_low_pass_filter(I,0.8,2)

S2 = cv2.Sobel(I2, -1, 1, 1, 1, 7)  
C2 = img_as_ubyte(I2)
C2 = cv2.Canny(C2, 100, 100) 

m=np.max(E)
E=E/(m+(m==0))
m=np.max(S)
S=S/(m+(m==0))
m=np.max(C)
C=C/(m+(m==0))
m=np.max(S1)
S1=S1/(m+(m==0))
m=np.max(C1)
C1=C1/(m+(m==0))
m=np.max(S2)
S2=S2/(m+(m==0))
m=np.max(C2)
C2=C2/(m+(m==0))

O=[I,E,S,C,S1,C1,S2,C2]
title=['input','RPT','sobel','canny','sobel after\naverage filter','canny after\naverage filter','sobel after\nbutterworth filter','canny after\nbutterworth filter']
figure,ax=plt.subplots(2,4)
for i in range(8):
	ax[i//4,i%4].set_title(title[i])
	ax[i//4,i%4].imshow(O[i],cmap='gray',vmin=0,vmax=1)
	ax[i//4,i%4].axis('off')
plt.show()

