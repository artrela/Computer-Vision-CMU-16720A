import os
import multiprocessing
from os.path import join, isfile

import numpy as np
import scipy.ndimage
import skimage.color
from PIL import Image
from sklearn.cluster import KMeans

# user imports
import scipy.spatial

def extract_filter_responses(opts, img):
    """
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    """

    filter_scales = opts.filter_scales # array [1,2]
    # ----- TODO -----
    
    #todo ================================== Ensure Num Channels ============================================
    
    # check the channels and adjust accordingly   
    img_channel = img.shape[-1]
    if img_channel == 3: #! normal
        pass
    elif img.ndim == 2: #! greyscale
        # stack the img on itself along the depth, or the last axis
        img = np.stack((img, img, img), axis=2)
    elif img_channel == 4: #! four channels
        # using indexing, take every value in the first two dims, but only the first
        # three channels
        img = img[:,:,2] 
    else:
        raise ValueError(f"Your image is of dimension {img_channel}, anticipated channels are 1, 3, or 4.")
    
    #todo ================================== Change to Lab Color Space ==================================
    
    img_lab = skimage.color.rgb2lab(img) # convert to lab color system
    
    #todo ================================== Streamline Filter Application ==============================
        
    def apply_filter(filter: str, scale: int, img_channels: tuple) -> np.ndarray:
        """ Apply one of four filters to each seperate channel and then into one image.

        Args:
            filter (str): Modes: Gaussian, Gaussian Laplace, Gaussian Derivate Along X, and Gaussian Derivative Along Y.
            Input is case sensitive.
            scale (int): Refers to the scale that the filter is subject to, corresponds to the size of the filter being applied.
            img_channels (tuple): Since each channel is filtered seperately and then combined, this tuple must be of len 2. 

        Returns:
            np.ndarray: Returns an np array of each channel that has been filtered seperately and then combined.
        """
        if filter == "Gaussian":
            # normalize vals, blurring effect
            r_filter = scipy.ndimage.gaussian_filter(img_channels[0], sigma=scale, order = 0)
            g_filter = scipy.ndimage.gaussian_filter(img_channels[1], sigma=scale, order = 0)
            b_filter = scipy.ndimage.gaussian_filter(img_channels[2], sigma=scale, order = 0)
        elif filter == "Gaussian Laplace":
            # notice sharp changes, 
            r_filter = scipy.ndimage.gaussian_laplace(img_channels[0], sigma=scale)
            g_filter = scipy.ndimage.gaussian_laplace(img_channels[1], sigma=scale)
            b_filter = scipy.ndimage.gaussian_laplace(img_channels[2], sigma=scale)
        elif filter == "Gaussian Derivative Along X":
            # deriv along x, vert line detect
            r_filter = scipy.ndimage.gaussian_filter(img_channels[0], sigma=scale, order = (0,1))
            g_filter = scipy.ndimage.gaussian_filter(img_channels[1], sigma=scale, order = (0,1))
            b_filter = scipy.ndimage.gaussian_filter(img_channels[2], sigma=scale, order = (0,1))
        elif filter == "Gaussian Derivative Along Y":
            # deriv along y, horiz line detect
            r_filter = scipy.ndimage.gaussian_filter(img_channels[0], sigma=scale, order = (1,0))
            g_filter = scipy.ndimage.gaussian_filter(img_channels[1], sigma=scale, order = (1,0))
            b_filter = scipy.ndimage.gaussian_filter(img_channels[2], sigma=scale, order = (1,0))
        
        # combine into one array
        rgb = [r_filter[..., np.newaxis], g_filter[..., np.newaxis], b_filter[..., np.newaxis]]
        # stack layers on each other
        stacked_channels = np.concatenate(rgb, axis=2)
        
        return stacked_channels
    
    #todo ================================== Apply Filter at Different Scales =============================== 
    # intitalize filter responses 
    filter_responses = []
    # seperate each channel to filter
    r = img_lab[:,:,0]
    g = img_lab[:,:,1]
    b = img_lab[:,:,2]
    # iterate through scales   
    for scale in filter_scales:
        gaus = apply_filter(filter="Gaussian", scale = scale, img_channels = (r, g, b))
        gaus_la = apply_filter(filter="Gaussian Laplace", scale = scale, img_channels = (r, g, b))
        gaus_der_x = apply_filter(filter="Gaussian Derivative Along X", scale = scale, img_channels = (r, g, b))
        gaus_der_y = apply_filter(filter="Gaussian Derivative Along Y", scale = scale, img_channels = (r, g, b))    

        if len(filter_responses) == 0:
            # if it is the first time through scales, can't add on top
            filter_responses = np.concatenate((gaus, gaus_la, gaus_der_x, gaus_der_y), axis=2)
        else:
            # add each filter to the responses by concantenated once a scale is completed
            filter_responses = np.concatenate((filter_responses, gaus, gaus_la, gaus_der_x, gaus_der_y), axis=2)
        
    
    return filter_responses


def compute_dictionary_one_image(args):
    """ Takes in an image path, opens the image, and save to a temporary file.

    Args:
        arg (tuple): In the form (opts, img_path)
    """
    # ----- TODO -----
    # extract params from args
    img_path = args[1]
    opts = args[0]
    
    # open the image and change to ndarray
    img = Image.open(join(opts.data_dir, img_path))
    img = np.array(img).astype(np.float32) / 255
    
    # obtain the filter response
    filter_response = extract_filter_responses(opts=opts, img=img) # filter_response.shape = (H, W, 3F)
    
    #fb_size = filter_response.shape[2] # needed to reshape the filter response with proper depth 
    # reshape filter_response to get a list of all pixels, with a depth the size of the filter bank (3 * filters * num scales)
    # this represents a list of all the pixels
    #fr_list = np.concatenate(filter_response, axis=-1)
    H = filter_response.shape[0] # height of fr
    W = filter_response.shape[1] # width of fr
    fbs = filter_response.shape[2] # how many filters we took
    fr_list = np.zeros(shape=(H*W, fbs)) # turns the image into a list 
    
    k = 0 # allows you to iterate through all pixels 
    for i in range(0, H):
        for j in range(0, W):
            fr_list[k] = filter_response[i][j][:]
            k += 1
            
    nPixels = fr_list.shape[0]
    # use alpha to get a subset of random pixels -> np.array of len alpha
    alpha_pixels = np.random.choice(nPixels, opts.alpha) # this will get a list of alpha pixels from the number of available pixels
    alpha_fr = fr_list[alpha_pixels] # get the response at the locations of alpha pixels
    
    temp_file_path = join(opts.feat_dir + "/" + img_path[0:-4] + ".npy")
    try:
        np.save(temp_file_path, alpha_fr)
    except FileNotFoundError:
        # if the file wasn't found then the img category folder isnt made, make one
        img_dir = img_path.split("/")[0]
        os.mkdir(join(opts.feat_dir + "/" + img_dir)) # go to feat dir and make new directory
        np.save(temp_file_path, alpha_fr)
    
    print(f"Finished saving {temp_file_path}...")
    return temp_file_path
    
    

def compute_dictionary(opts, n_worker=1):
    """
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    """
    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    # ----- TODO -----
    # list of params to pass to compute_dictionary_one_img: unique training file path, same feat_dict path
    pool_params = [(opts, img_path) for img_path in train_files] 
    with multiprocessing.Pool(processes=n_worker) as pool:
        results = pool.map(compute_dictionary_one_image, pool_params) # map all the image files to a process
    pool.close() # close the pool
    print("Finished extracting filter all responses.")
    
    # load back up all the npy files into a list of filter responses
    print("Loading all temporary files...")
    filter_responses = []
    for temp_file in results:
        print(f"Loading {temp_file}...")
        fr_img = np.load(temp_file)
        # list of fr size (image, alpha, filters)
        filter_responses.append(fr_img)
    
    # change to np.array to resize 
    #filter_responses = np.concatenate(filter_responses, axis=0)
    T = len(filter_responses) # num images
    #alpha = filter_responses.shape[1] # also opts.alpha
    fb_size = 3 * len(opts.filter_scales) * 4 # filter bank size
    alphaT = opts.alpha * T
    # resize to (alpha*t) x 3F
    alpha_fr = np.zeros(shape=(alphaT, fb_size))
    k = 0
    for i in range(0, T):
        for j in range(0, opts.alpha):
            alpha_fr[k] = filter_responses[i][j]
            k += 1
    
    print("Loaded all filter responses...") 
    # calculate the kmeans and save the dictionary
    print("Started calculating K-means...")
    kmeans = KMeans(n_clusters=K).fit(alpha_fr)
    dictionary = kmeans.cluster_centers_
    print("Finished calculating K-means, saving cluster centers...")
    # example code snippet to save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), dictionary)


def get_visual_words(opts, img, dictionary):
    """
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    * dictionary : numpy.ndarray of shape (K,3F)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    """

    # ----- TODO -----
    feature_responses = extract_filter_responses(opts, img)
    
    fr_H = feature_responses.shape[0]
    fr_W = feature_responses.shape[1]
    feature_bank = feature_responses.shape[2]
    
    # now we have a list of pixel responses across 36 channels
    fr_list = np.zeros(shape=(fr_H*fr_W, feature_bank))
    k = 0 # allows you to iterate through all pixels 
    for i in range(0, fr_H):
        for j in range(0, fr_W):
            fr_list[k] = feature_responses[i][j][:]
            k += 1
    
    # for each pixel, finds the distance to a cluster center across 36 channels    
    dists =  scipy.spatial.distance.cdist(fr_list, dictionary)
    # for each pixel in dist, we have 10 distances, find the min dist
    # this min dist shows which cluster the image belongs to 
    word_map = np.argmin(dists, axis=1)
    
    word_map = word_map.reshape(fr_H, fr_W, -1)
    
    return word_map