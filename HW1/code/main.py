 
from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts

# user imports 
import math
import random

def main():
    opts = get_opts() 
    
    #//Q1.1
    #// join the path where the image is > open the image > change its type to float32 
    #// > dividing by 255 puts all values between 0-1
    # img_path = join(opts.data_dir,  'aquarium/sun_aztvjgubyrgvirup.jpg')
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32) / 255
    # filter_responses = visual_words.extract_filter_responses(opts, img)
    # util.display_filter_responses(opts, filter_responses)
    
#// Q1.2
    n_cpu = util.get_num_CPU()
    visual_words.compute_dictionary(opts, n_worker=n_cpu)

    # Q1.3
    # img_3 = ["kitchen/sun_aaqhazmhbhefhakh.jpg", "desert/sun_aaqyzvrweabdxjzo.jpg", "windmill/sun_bxpvmlxprftjwynu.jpg"] 
    # img = img_3[2]
    # # for img in img_3:
    # img_path = join(opts.data_dir, img)
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32)/255
    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    # wordmap = visual_words.get_visual_words(opts, img, dictionary)
    #     util.visualize_wordmap(wordmap)
    #     util.visualize_wordmap(img)
    
    # Q2.1-2.4
    # n_cpu = util.get_num_CPU()
    visual_recog.build_recognition_system(opts, n_worker=n_cpu)
        
    #Q2.5
    #n_cpu = util.get_num_CPU()
    conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)

    print("\nConfusion Matrix: \n", np.matrix(conf), "\n\n", "Final Accuracy: ", accuracy)

    np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')
    
    #! Create a log file to write information on parameter testing
    # with open("parameter_log.txt", "a") as f:
    #     f.write(f"Layers: {opts.L} | Alpha: {opts.alpha} | Scales: {opts.filter_scales} | K: {opts.K} | Accuracy: {accuracy}\n")
    #     f.close()
        
       

if __name__ == '__main__':
    main()


