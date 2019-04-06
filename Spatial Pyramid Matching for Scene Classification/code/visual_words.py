import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import util
import random
import skimage.io

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''

    if len(image.shape) == 2: # if image has single-channel
        image = np.tile(image[:, newaxis], (1, 1, 3)) # Make it three similar channel

    if image.shape[2] == 4:   # if image has 4 channels 
        image = image[:,:,0:3]  # omit the last channel

    image = skimage.color.rgb2lab(image) # Converting to LAB scale
    scales = [1,2,4,8,8*np.sqrt(2)]
    for i in range(len(scales)):
        for c in range(3):
            #img = skimage.transform.resize(image, (int(ss[0]/scales[i]),int(ss[1]/scales[i])),anti_aliasing=True)
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i])
            if i == 0 and c == 0:
                imgs = img[:,:,np.newaxis]
            else:
                imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_laplace(image[:,:,c],sigma=scales[i])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[0,1])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
        for c in range(3):
            img = scipy.ndimage.gaussian_filter(image[:,:,c],sigma=scales[i],order=[1,0])
            imgs = np.concatenate((imgs,img[:,:,np.newaxis]),axis=2)
    #print("Extract Filter Responses Run")
    return imgs

def get_visual_words(image,dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    filter_response = extract_filter_responses(image)
        
    width   =image.shape[1]      # the width of the image
    height  =image.shape[0]      # the height of the image

    filter_response2D=filter_response.reshape(width*height,-1) # response is reshaped for distance function
    distance=scipy.spatial.distance.cdist(filter_response2D,dictionary)

    min_distance=np.argmin(distance,axis=1)

    wordmap=min_distance.reshape(image.shape[0],image.shape[1])

    return wordmap


def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file
    * time_start: time stamp of start time

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)
    '''
    #print("Single Image is processing")
    i,alpha,image_path = args
    # ----- TODO -----
    #~pass
    #print(i)
    #print(alpha)
    #print(image_path)
    #print("Single Image is processing")
    path_img = "../data/"+image_path
    image = skimage.io.imread(path_img)
    image = image.astype('float')/255
    filter_responses = extract_filter_responses(image)
    x=np.random.randint(0,image.shape[0],alpha)
    y=np.random.randint(0,image.shape[1],alpha)

    PixelResponse=filter_responses[x,y,:]

    np.savez("../temp/"+str(i)+".npz", FilterResponses=PixelResponse)

    return None

def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel
    
    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    '''

    train_data = np.load("../data/train_data.npz")
    # ----- TODO -----

    
    K = 150
    alpha = 100

    os.makedirs("../temp/", exist_ok=True)
    for i in range(len(train_data['files'])):
        args=[i,alpha,train_data['files'][i]]
        compute_dictionary_one_image(args)

    FilterResponses = np.empty((0, 60))

    for i in range(train_data['files'].shape[0]):
        PixelResponse = np.load("../temp/"+str(i)+".npz")
        FilterResponses = np.append(FilterResponses, PixelResponse['FilterResponses'], axis=0)


    kmeans = sklearn.cluster.KMeans(n_clusters=K,n_jobs=num_workers).fit(FilterResponses)
    dictionary = kmeans.cluster_centers_

    print("compute_dictionary is done")
    np.save("dictionary.npy", dictionary)



