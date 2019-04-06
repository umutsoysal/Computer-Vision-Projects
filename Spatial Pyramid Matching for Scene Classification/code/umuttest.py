
# Template for the assignment

# Libraries
import numpy as np 
import multiprocessing
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import util
import random
import torchvision
import matplotlib.pyplot as plt
#import matplotlib 
#matplotlib.use('TkAgg')
import visual_words
import skimage.io


# Number of cores 
print(multiprocessing.cpu_count())

train_data = np.load("../data/train_data.npz")

#print(train_data.shape)

#lst = train_data.files
#%for item in lst:
#    print(item)
#    print(train_data[item])
    
i=60
alpha=100
print(train_data['files'][i])
path_img = "../data/"+train_data['files'][i]
image = skimage.io.imread(path_img)
image = image.astype('float')/255
filter_responses = visual_words.extract_filter_responses(image)
#util.display_filter_responses(filter_responses)

RandomPoints=np.random.permutation(alpha)

x=np.random.randint(0,image.shape[0],alpha)
y=np.random.randint(0,image.shape[1],alpha)

#RandomPoints=zip(x,y)

PixelResponse=filter_responses[x,y,:]

#np.savez("../temp/"+str(i)+".npz", FilterResponses=PixelResponse)


