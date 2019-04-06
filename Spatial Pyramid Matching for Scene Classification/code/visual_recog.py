import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
import multiprocessing
import skimage.io


def compute_feature_one_image(args):
    i, image_path, label = args
    dictionary = np.load("dictionary.npy")

    SPM_layer_num = 2
    K = dictionary.shape[0]

    #print(i)
    #print(image_path)
    #print(label)

    #print("DELETETHIS",get_image_feature("../data/" + image_path, dictionary, SPM_layer_num, K))
    feature = np.reshape(get_image_feature("../data/" + image_path, dictionary, SPM_layer_num, K), (1, -1))
    #print("Features of a one image",feature)
    np.savez("../temp/"+str(i)+".npz", feature=feature, label=label)



def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''


    train_data = np.load("../data/train_data.npz")  # Load the training data
    dictionary = np.load("dictionary.npy")   # Load the dictionary
    # ----- TODO -----

    SPM_layer_num = 2
    n_word = dictionary.shape[0]
    n_training_sample= train_data['files'].shape[0]


    features = np.empty((0, int(n_word*(4**(SPM_layer_num+1) - 1)/3)))
    #print("DEBUG",features)
    labels = []

    os.makedirs("../temp/", exist_ok=True)

    #for i in range(2): for testing
    for i in range(train_data['files'].shape[0]):
       args=[i, train_data['files'][i], train_data['labels'][i]]
       #args = zip(list(range(n_training_sample)), train_data['files'], train_data['labels']) 
       
       #uncomment this if there are no temp file
       #compute_feature_one_image(args)
       #uncomment this if there are no temp file before submission
    

    #with multiprocessing.Pool(num_workers) as p:
    #    args = zip(list(range(n_training_sample)), train_data['files'], train_data['labels'])
    #    p.map(compute_feature_one_image, args)

    for i in range(train_data['files'].shape[0]):
        temp = np.load("../temp/"+str(i)+".npz")  # load the temp file we create
        #print("what is in",temp.files)
        #print(i)
        #print(temp['feature'])
        features = np.append(features, np.reshape(temp['feature'], (1,-1)), axis=0)
        
        labels = np.append(labels, temp['label'])

    np.savez("trained_system.npz", features=features, labels=labels, dictionary=dictionary, SPM_layer_num=SPM_layer_num)



def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''
    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system.npz")

    features = trained_system['features']
    labels = trained_system['labels']
    dictionary = trained_system['dictionary']
    SPM_layer_num = int(trained_system['SPM_layer_num'])
    # ----- TODO -----
    test_sample_num = test_data['files'].shape[0]

    conf = np.zeros((8,8))

    for i in range(len(test_data['files'])): 
        img_path = "../data/"+test_data['files'][i] # Read the test image
        print("Evaluate recognition",img_path)
        image = skimage.io.imread(img_path) #     load the image
        wordmap = visual_words.get_visual_words(image, dictionary) # Create the wordmap
        hist = get_feature_from_wordmap_SPM(wordmap, SPM_layer_num, dictionary.shape[0]) # Crate the histogram of dictionary
        sim = distance_to_set(hist, features) # Calculate the distances to each scene type
        #print("Similarity",sim)
        predicted_label = trained_system['labels'][np.argmax(sim)] # Find the corresponding scene class
        print("Prediction", predicted_label)
        conf[test_data['labels'][i], int(predicted_label)] += 1

            

    accuracy=np.trace(conf)/np.sum(conf)
    print("Prediction Accuracy of Bag of Words Model:", accuracy)
    return conf, accuracy



def get_image_feature(file_path,dictionary,layer_num,K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    #print("test"+file_path)
    image = skimage.io.imread(file_path)
    wordmap = visual_words.get_visual_words(image,dictionary)
    hist_all= get_feature_from_wordmap_SPM(wordmap, layer_num, K)
    #print("Get image Feature output size",hist_all.shape)
    return hist_all
    # ----- TODO -----


def distance_to_set(word_hist,histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    
    # ----- TODO -----
    # the minimum value in each bin for both histograms
    intersection=np.minimum(word_hist,histograms)
    sim=np.sum(intersection,axis=1)
    return sim 


def get_feature_from_wordmap(wordmap,dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    
    # ----- TODO -----
    # Histogram is normalized over 1 with density=True option
    hist=np.histogram(wordmap,bins=dict_size,density=True)
    return hist


def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    # ----- TODO -----



    hist_all = []
    norm_factor = wordmap.shape[0]*wordmap.shape[1]

    
    # This code assumes layer num is 2 as it is requested in the assignment
    for i in range(layer_num+1):
        if i == 0 or i == 1: 
            weight =2**(-layer_num) # Special condition for the layer 0 and 1
        else:
            weight =2**(layer_num-i-1)
        n_cell=2**i

        wordmap_rows = np.array_split(wordmap, n_cell, axis=0)
        for rows in wordmap_rows:
            wordmap_cols = np.array_split(rows, n_cell, axis=1)
            for cols in wordmap_cols:
                hist, bin_edges = np.histogram(cols, bins=dict_size,density=True)
                #print("Inter hists",hist.shape)
                hist_all = np.append(hist_all, hist*weight)
                #print("Appended hists",hist_all.shape)

    #print("Get image Feature from wordmap SPM size",hist_all.shape)
    #print(i)
    return hist_all






    

