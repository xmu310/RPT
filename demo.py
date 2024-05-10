import sys
import numpy as np
from skimage import io,color,transform
from matplotlib import pyplot as plt
from runReal import runReal
I=io.imread(sys.argv[1])/255
if I.ndim==3:
	I=color.rgb2gray(I)
I=transform.resize(I,(65,65))
E=runReal(I)
m=np.max(E)
E=E/(m+(m==0))
O=[I,E]
title=['input','output']
figure,ax=plt.subplots(1,2)
for i in range(2):
	ax[i].set_title(title[i])
	ax[i].imshow(O[i],cmap='gray',vmin=0,vmax=1)
	ax[i].axis('off')
plt.show()

