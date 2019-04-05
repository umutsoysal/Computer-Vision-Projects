import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
from LucasKanade import *
from LucasKanadeBasis import *
from time import sleep
from scipy.interpolate import RectBivariateSpline
import scipy.io as sio

print('Downloading SYLV dataset...')
bases = np.load('../data/sylvbases.npy')
data=np.load('../data/sylvseq.npy')
print('Downloaded')
print(data.shape)


rect=[101,61,155,107] # initial position of the rectange
rect1=[101,61,155,107] # initial position of the rectange


width=rect[2]-rect[0]
height=rect[3]-rect[1]

p0=np.array([0,0])
p1=np.array([0,0])
frameNumber=data.shape[2]


rectsArray=np.zeros([frameNumber,4])
rectsArray[0,:]=rect

rectsArray1=np.zeros([frameNumber,4])
rectsArray1[0,:]=rect1


fig=plt.figure(1)
ax=fig.add_subplot(111)
ax.set_title("LucasKanade Tracking with Single Template")


for i in range(data.shape[2]-1):  # iteration through all the frames.
	image=data[:,:,i]
	image2=data[:,:,i+1]
	p0=np.array([0,0]) #initial value for LK algorithm 	
	p0=LucasKanade(image,image2,rect,p0) # output of the LK algorithm
	rect[0]=rect[0]+(p0[1]) #y 
	rect[1]=rect[1]+(p0[0])
	rect[2]=rect[2]+(p0[1])
	rect[3]=rect[3]+(p0[0])
	rectsArray[i,:]=rect
	width=rect[2]-rect[0]
	height=rect[3]-rect[1]
	p1 = LucasKanadeBasis(image, image2, rect1, bases)
	rect1[0]=rect1[0]+(p1[1]) #y 
	rect1[1]=rect1[1]+(p1[0])
	rect1[2]=rect1[2]+(p1[1])
	rect1[3]=rect1[3]+(p1[0])
	rectsArray1[i,:]=rect1
	rectangle = patches.Rectangle((rect[0],rect[1]),width,height,linewidth=2,edgecolor='y',facecolor='none')
	rectangle1 = patches.Rectangle((rect1[0],rect1[1]),width,height,linewidth=2,edgecolor='b',facecolor='none')
	ax.add_patch(rectangle) # add the rectangle patch onto image
	ax.add_patch(rectangle1) # add the rectangle patch onto image
	plt.imshow(image2,cmap='gray')
	plt.pause(0.001)
	if i in [1, 100, 200, 300, 400]:
		plt.savefig("ApperanceBasis"+str(i)+".png")  
	ax.clear()

print("image flow is")
print(p0)
np.save("sylvseqrects.npy",rectsArray1)