import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import *
from time import sleep
from scipy.interpolate import RectBivariateSpline
import scipy.io as sio

# write your script here, we recommend the above libraries for making your animation
print('Downloading CARSEQ dataset...')
data=np.load('../data/carseq.npy')
print('Downloaded')
print(data.shape)

rect=[59,116,145,151] 
rect0=[59,116,145,151] 
rect2=[59,116,145,151]
width=rect[2]-rect[0]
height=rect[3]-rect[1]

p0=np.array([0,0])
frameNumber=data.shape[2]



rectsArray=np.zeros([frameNumber,4])
rectsArray[0,:]=rect

rectsArray2=np.zeros([frameNumber,4])
rectsArray2[0,:]=rect2

fig=plt.figure(1)
ax=fig.add_subplot(111)
ax.set_title("LucasKanade Tracking with Template Correction")



for i in range(data.shape[2]-1):  # iteration through all the frames.
    
    image=data[:,:,i]
    image2=data[:,:,i+1]
    image0=data[:,:,0] # First frame
  
  
    p0=np.array([0,0]) #initial value for LK algorithm  
    p0=LucasKanade(image,image2,rect,p0) # output of the LK algorithm
    
    p1 = np.array((rect[1] + p0[0] - rect0[1], rect[0] + p0[1] - rect0[0])).reshape(2)
    

    p2 = LucasKanade(image0, image2,rect0, p1)
    


    #finding the coordinates of the new rectangle
    rect[0]=rect[0]+(p0[1]) #y 
    rect[1]=rect[1]+(p0[0])
    rect[2]=rect[2]+(p0[1])
    rect[3]=rect[3]+(p0[0])
    #print(rect)
    rectsArray[i,:]=rect
    width=rect[2]-rect[0]
    height=rect[3]-rect[1]
    #print(width)
    #print(height)
    #if (i==2 or i%100==0):
    
    if np.linalg.norm(p2 - p1) < 1:
        rect2[0] += p0[1]
        rect2[1] += p0[0]
        rect2[2] = rect[0] + width
        rect2[3] = rect[1] + height
    else:
        rect2[0] = rect0[0] + p2[1]
        rect2[1] = rect0[1] + p2[0]
        rect2[2] = rect2[0] + width
        rect2[3] = rect2[1] + height

    rectsArray2[i,:]=rect2


    # Display of tracking
    rectangle = patches.Rectangle((rect[0],rect[1]),width,height,linewidth=2,edgecolor='y',facecolor='none')
    rectangle2 = patches.Rectangle((rect2[0],rect2[1]),width,height,linewidth=2,edgecolor='g',facecolor='none')
    
    ax.add_patch(rectangle) # add the rectangle patch onto image
    ax.add_patch(rectangle2) # add the rectangle patch onto image
    
    plt.imshow(image2,cmap='gray')

    plt.pause(0.001) 
    if i in [1, 100, 200, 300, 400]:
        plt.savefig("CarWithTemplateCorrection"+str(i)+".png") 
    ax.clear()
        #plt.savefig('Tracking.png')
print("image flow is")
print(p0)
np.save("carseqrects-wcrt.npy",rectsArray2)