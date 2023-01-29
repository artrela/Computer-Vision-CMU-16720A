import string
import pickle
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from nn import *
import cv2

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']


if False:  # view the data
    np.random.shuffle(train_x)
    for crop in train_x:
        plt.imshow(crop.reshape(32, 32).T, cmap="Greys")
        plt.show()

max_iters = 50
# pick a batch size, learning rate
batch_size = 64
learning_rate = 2e-3
hidden_size = 64
##########################
##### your code here #####
##########################


for i in range(3):
    cv2.imshow("", np.reshape(train_x[i, :], (32, 32)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


batches = get_random_batches(train_x, train_y, batch_size)
batch_num = len(batches)

params = {}

# initialize layers
initialize_weights(train_x.shape[1], hidden_size, params, "layer1")
initialize_weights(hidden_size, train_y.shape[1], params, "output")
layer1_W_initial = np.copy(params["Wlayer1"])  # copy for Q3.3

train_loss = []
valid_loss = []
train_acc = []
valid_acc = []
for itr in range(max_iters):
    # record training and validation loss and accuracy for plotting
    h1 = forward(train_x, params, 'layer1')
    probs = forward(h1, params, 'output', softmax)
    loss, first_acc = compute_loss_and_acc(train_y, probs)
    train_loss.append(loss/train_x.shape[0])
    train_acc.append(first_acc)
    h1 = forward(valid_x, params, 'layer1')
    probs = forward(h1, params, 'output', softmax)
    loss, acc = compute_loss_and_acc(valid_y, probs)
    valid_loss.append(loss/valid_x.shape[0])
    valid_acc.append(acc)

    total_loss = 0
    avg_acc = 0
    for xb, yb in batches:
        # forward
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        # loss
        # be sure to add loss and accuracy to epoch totals
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        avg_acc += acc
        # backward
        delta1 = probs - yb
        delta2 = backwards(delta1, params, 'output', linear_deriv)
        backwards(delta2, params, 'layer1', sigmoid_deriv)

        for k, v in sorted(list(params.items())):
            if 'grad' in k:
                name = k.split('_')[1]
                # val = val + lr * val_dot
                params[name] -= learning_rate * params[k]

    avg_acc /= len(batches)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(
            itr, total_loss, first_acc))

    # if itr % 100 == 0:
    #     learning_rate *= 0.9

# record final training and validation accuracy and loss
h1 = forward(valid_x, params, 'layer1')
probs = forward(h1, params, 'output', softmax)
loss, acc = compute_loss_and_acc(valid_y, probs)
valid_loss.append(loss/valid_x.shape[0])
valid_acc.append(acc)

h1 = forward(train_x, params, 'layer1')
probs = forward(h1, params, 'output', softmax)
loss, acc = compute_loss_and_acc(train_y, probs)
train_loss.append(loss/train_x.shape[0])
train_acc.append(acc)


# report validation accuracy; aim for 75%
print('Validation accuracy: ', valid_acc[-1])

# compute and report test accuracy
h1 = forward(test_x, params, 'layer1')
test_probs = forward(h1, params, 'output', softmax)
_, test_acc = compute_loss_and_acc(test_y, test_probs)
print('Test accuracy: ', test_acc)

# save the final network
saved_params = {k: v for k, v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# plot loss curves
plt.plot(range(len(train_loss)), train_loss, label="training")
plt.plot(range(len(valid_loss)), valid_loss, label="validation")
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(train_loss)-1)
plt.ylim(0, None)
plt.legend()
plt.grid()
plt.show()

# plot accuracy curves
plt.plot(range(len(train_acc)), train_acc, label="training")
plt.plot(range(len(valid_acc)), valid_acc, label="validation")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.xlim(0, len(train_acc)-1)
plt.ylim(0, None)
plt.legend()
plt.grid()
plt.show()


# Q3.3

# visualize weights
fig = plt.figure(figsize=(8, 8))
plt.title("Layer 1 weights after initialization")
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.05)
for i, ax in enumerate(grid):
    ax.imshow(layer1_W_initial[:, i].reshape((32, 32)).T, cmap="Greys")
    ax.set_axis_off()
plt.show()

v = np.max(np.abs(params['Wlayer1']))
fig = plt.figure(figsize=(8, 8))
plt.title("Layer 1 weights after training")
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.05)
for i, ax in enumerate(grid):
    ax.imshow(params['Wlayer1'][:, i].reshape(
        (32, 32)).T, cmap="Greys", vmin=-v, vmax=v)
    ax.set_axis_off()
plt.show()

# Q3.4
confusion_matrix = np.zeros((test_y.shape[1], test_y.shape[1]))

# compute confusion matrix
##########################
##### your code here #####
##########################

for i in range(0, test_y.shape[0]):
    # find the current row's prediction and ground truth label
    pred = np.argmax(test_y[i, :])
    true = np.argmax(test_probs[i, :])

    confusion_matrix[pred, true] += 1

plt.imshow(confusion_matrix, interpolation='nearest')
plt.grid()
plt.xticks(np.arange(36),
           string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),
           string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.xlabel("predicted label")
plt.ylabel("true label")
plt.colorbar()
plt.show()
