import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b.
# X be [Examples, Dimensions]


def initialize_weights(in_size, out_size, params, name=''):
    W, b = None, None

    ##########################
    ##### your code here #####
    ##########################

    var = np.sqrt(6 / (in_size + out_size))

    W = np.random.randn(in_size, out_size) * var
    b = np.zeros(shape=out_size)

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function


def sigmoid(x):

    # Simply the equation for a sigmoid activation function
    res = 1 / (1 + np.exp(-x))

    return res

############################## Q 2.2.1 ##############################


def forward(X, params, name='', activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    ##########################
    ##### your code here #####
    ##########################

    """
    1. Forward prop without activation: W.T * X + b
    2. Pass through the activation function
    """

    pre_act = np.dot(X, W) + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row


def softmax(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################

    """
    1. For numerical stability, shift all values over so the maximum is now 0
        - expand the dims so you can broadcast, need a singleton dimension
    2. Exponentiate all the xvals
    3. Take a cumulative sum acros the rows, so you can softmax each example
    4. Get the softmax result to make a probability vector
        - Gives prob. that a particular input is of a particular class
    """

    # 1.
    x_shift = x - np.expand_dims(np.amax(x, axis=1), axis=1)
    # 2.
    x_exp = np.exp(x_shift)
    # 3.
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    # 4.
    res = x_exp / x_sum

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]


def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    ##########################
    ##### your code here #####
    ##########################

    """
    1. Use cross-entropy loss function
    2. Find which label was accurately guessed
    """

    # 1.
    loss = - np.sum((y * np.log(probs)))
    # 2.
    prob_labels = np.argmax(probs, axis=1)
    y_labels = np.argmax(y, axis=1)
    acc = (y_labels == prob_labels).astype(int)
    acc = np.sum(acc) / acc.shape[0]

    return loss, acc

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act


def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res


def backwards(delta, params, name='', activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # (don't forget activation_deriv is a function of post_act)
    # then compute the derivative W, b, and X
    ##########################
    ##### your code here #####
    ##########################

    grad_A = delta * activation_deriv(post_act)
    grad_X = np.dot(grad_A, W.T)
    grad_W = np.dot(X.T, grad_A)
    grad_b = np.sum(grad_A, axis=0)

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]


def get_random_batches(x, y, batch_size):
    batches = []
    ##########################
    ##### your code here #####
    ##########################

    # shuffled the data
    shuffled_x, shuffled_y = shuffled_data(x, y)
    # get the number of batches
    num_batches = x.shape[0] / batch_size
    # split the batches into number of batches
    xbatches = np.split(shuffled_x, num_batches)
    ybatches = np.split(shuffled_y, num_batches)
    # for the splot arrays, zip them together, then append them to batches
    for xbatch, ybatch in zip(xbatches, ybatches):
        batches.append((xbatch, ybatch))

    return batches


def shuffled_data(x, y):
    shuffled_ind = np.random.permutation(x.shape[0])
    shuffled_x = x[shuffled_ind, :]
    shuffled_y = y[shuffled_ind, :]
    return shuffled_x, shuffled_y
