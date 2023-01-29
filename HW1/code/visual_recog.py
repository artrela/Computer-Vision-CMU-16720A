from ctypes import LibraryLoader
from lib2to3.pgen2.literals import test
import os
import math
import multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words


def get_feature_from_wordmap(opts, wordmap, img_path):
    """
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    """

    K = opts.K
    # ----- TODO -----
    
    hist = np.zeros(shape=(K)) # init hist size
    
    # resize wordmap to a 1D array
    wm_H = wordmap.shape[0]
    wm_W = wordmap.shape[1]
    wordmap_list = np.resize(wordmap, new_shape=(wm_H*wm_W))

    # go through clusters (1-10) and count how many times it appears in the wordmap
    for cluster in range(0, K): 
        occur = np.count_nonzero(wordmap_list == cluster, axis = 0)
        hist[cluster-1] = occur # 
    
    # # perform l1 normalization on the histogram entries
    L1_norm = np.linalg.norm(hist, ord=1)

    # # normalize hist
    # # this now represents the freq of which features occur
    hist = hist/L1_norm
        
    return hist

def get_feature_from_wordmap_SPM(opts, wordmap, img_path):
    """
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape K*(4^(L+1) - 1) / 3
    """

    K = opts.K
    L = opts.L
    # ----- TODO -----
    
    #todo =============================== find cells for each layer =====================================
    
    layers = [] # create a layers array to hold each layers cells
    for i in range(0,  L + 1):
        # find and floor the number of cells in each row - will tell you how many pixels to jump below 
        cell_W = int(wordmap.shape[1] // (2 ** i))
        cell_H = int(wordmap.shape[0] // (2 ** i))
        # calculate the cells for the finest layer 
        cells = []
        
        for x in range(0,wordmap.shape[0],cell_H):
            for y in range(0,wordmap.shape[1],cell_W):
                    if len(cells) < (2**i * 2**i):
                        cell = wordmap[x:x+cell_H,y:y+cell_W, :]
                        if cell.shape[0] == cell_H and cell.shape[1] == cell_W:
                            cells.append(np.array(cell))
           
        layers.append(cells)
   
    #todo ============================= get feature maps =============================================   
    
    list_hists = []
    for layer in layers:
        layer_hist = []
        
        for cell in layer:
            cell_hist = get_feature_from_wordmap(opts, cell, img_path)
            layer_hist.append(cell_hist)
        
        layer_hist = np.concatenate(layer_hist, axis = 0)
        list_hists.append(layer_hist)          
    
    
    #todo =============================== obtain layer weights ============================================= 
    weights = []
    for layer in range(0, L + 1):
        if layer == 0 or layer == 1:
            weight = 2 ** -L
        else:
            weight = 2**(layer-L-1)
        weights.append(weight)
        
    #todo ============================== get total histogram ======================================== 
    
    # apply the weights 
    for i in range(0, len(list_hists) - 1):
        list_hists[i] = weights[i] * list_hists[i]
    
    hist_all = np.concatenate(list_hists, axis = 0)
    
    L1_norm = np.linalg.norm(hist_all, ord=1) 
    
    hist_all = hist_all/L1_norm
    
    return hist_all

def get_image_feature(opts, img_path, dictionary):
    """
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    """

    # ----- TODO -----
    
    #load the img
    img = Image.open(join(opts.data_dir, img_path))
    img = np.array(img).astype(np.float32) / 255
    # # get the wordmap from the image
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    # use the visual words to get the SPM features
    feature = get_feature_from_wordmap_SPM(opts, wordmap, img_path)
  
    print(f"Obtaining features for images...")
    
    return feature 


def build_recognition_system(opts, n_worker=1):
    """
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    """
    # 
    
    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, "train_labels.txt"), np.int32)
    dictionary = np.load(join(out_dir, "dictionary.npy"))

    # ----- TODO -----
       
    # all we need to do is get the features from SPM
    # need to obtain the image features
    pool_params = [(opts, img_path, dictionary) for img_path in train_files] 
    with multiprocessing.Pool(processes=n_worker) as pool:
        results = pool.starmap(get_image_feature, pool_params) # can take multiple arguments
    pool.close()
    
    # results is a list of N hists, convert to a matrix
    features = np.stack(results, axis = 0)
    print("Done Getting Features. Shape: ", features.shape)
    # example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )


def similarity_to_set(word_hist, histograms):
    """
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    """

    # ----- TODO -----
    
    # find the minimum betweeen an individual word_hist and the N list of histograms
    # this will refer to the overlap between the word_hist and each histogram accross the K dimension
    mins = np.minimum(word_hist, histograms)
    
    # sum those minimum values to find the similarity
    # higher values indicate a higher change of similarity
    sim = np.sum(mins, axis = 1)
    
    return sim


def evaluate_recognition_system(opts, n_worker=1):
    """
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, "trained_system.npz"))
    dictionary = trained_system["dictionary"]

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system["SPM_layer_num"]

    test_files = open(join(data_dir, "test_files.txt")).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, "test_labels.txt"), np.int32)

    # ----- TODO -----
    
    
    # get a histogram for each test_file
    pool_params = [(opts, img_path, dictionary) for img_path in test_files] 
    with multiprocessing.Pool(processes=n_worker) as pool:
        test_hists = pool.starmap(get_image_feature, pool_params) # can take multiple arguments
    pool.close()
    
    # all the histograms of trained system
    trained_hists = trained_system["features"]
    # all the labels of the trained images
    train_labels = np.loadtxt(join(data_dir, "train_labels.txt"), np.int32)
    
    conf = np.zeros(shape=(8,8))
    correct = 0
    accuracy = 0
    for i in range(0, len(test_hists) - 1):
        sim = similarity_to_set(test_hists[i], trained_hists) # find the similarity vector, len = trained_hists
        max_sim = max(sim) # find the max similarity score
        max_loc = np.where(sim == max_sim) # location of max similarity
        # max_sim is the hist the test val is most like > get its position in the trained_hists list > map that to a label
        predicted_label = train_labels[max_loc[0][0]]
        
        if predicted_label == test_labels[i]:
            correct += 1 
        
        accuracy = round((correct/(i+1))*100, 2)
        
        print(f"Expected Result: {test_labels[i]} | Predicted Result: {predicted_label} | Percent Complete: {round(((i+1)/len(test_hists))*100, 2)}% | Accuracy: {accuracy}%")
        
        # add to conf matrix: predicted along axis 0, actual along axis 1
        conf[predicted_label][test_labels[i]] += 1
        
    accuracy = np.trace(conf)/np.sum(conf)
   
    return conf, accuracy
    
    
    
def compute_IDF(opts, n_worker=1):
    # YOUR CODE HERE
    pass

def evaluate_recognition_System_IDF(opts, n_worker=1):
    # YOUR CODE HERE
    pass
    
    