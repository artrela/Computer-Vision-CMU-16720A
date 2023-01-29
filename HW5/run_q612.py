import string
import pickle
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from nn import *
import sklearn.metrics
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

# layer dims
examples = train_x.shape[0]
input_size = train_x.shape[1]
classes = train_y.shape[1]

# for the y data, change to examples x 1
train_y = np.argmax(train_y, 1)
valid_y = np.argmax(valid_y, 1)
test_y = np.argmax(test_y, 1)

# Generate the tensors from np arrays
train_x = torch.from_numpy(train_x).to(torch.float32)
train_y = torch.from_numpy(train_y)

valid_x = torch.from_numpy(valid_x).to(torch.float32)
valid_y = torch.from_numpy(valid_y)

test_x = torch.from_numpy(test_x).to(torch.float32)
test_y = torch.from_numpy(test_y)

# Create a tensor dataset object so we can create a dataloader object
train_ds = TensorDataset(train_x, train_y)
valid_ds = TensorDataset(valid_x, valid_y)
test_ds = TensorDataset(test_x, test_y)

# Create a data loader object for the loop
train_data = DataLoader(train_ds, batch_size=32,
                        shuffle=True)
valid_data = DataLoader(valid_ds, batch_size=32,
                        shuffle=True)
test_data = DataLoader(test_ds, batch_size=32,
                       shuffle=True)

# establish params
max_iters = 10
# pick a batch size, learning rate
batch_size = 64
learning_rate = 2e-3
hidden_size = 150

model = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=1, out_channels=7, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Conv2d(in_channels=7, out_channels=13, kernel_size=5),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Flatten(),
    torch.nn.Linear(325, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, classes),
    torch.nn.LogSoftmax(dim=1))

loss_funct = nn.NLLLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

nn_params = {"training acc": 0,
             "training loss": 0,
             "validation acc": 0,
             "validation loss": 0,
             "testing acc": 0,
             "testing loss": 0
             }

nn_output = np.zeros(shape=(6, max_iters))

for i in range(max_iters):

    # =============== data training ====================
    iter_acc = 0
    iter_loss = 0

    for batch_train in train_data:

        batch_x, batch_y = batch_train

        batch_x = batch_x.view(-1, 1, 32, 32)

        optimizer.zero_grad()

        y_pred = model(batch_x)

        loss = loss_funct(y_pred, batch_y)
        iter_loss = loss.item()

        loss.backward()

        optimizer.step()

        _, predicted = torch.max(y_pred.data, 1)
        iter_acc += torch.sum(predicted == batch_y).item()

    iter_acc /= examples

    nn_params["training acc"] = (nn_params['training acc'] + iter_acc) / 2
    nn_params['training loss'] = iter_loss

    # ============================ validation data ===========
    iter_acc = 0
    iter_loss = 0

    valid_x = valid_x.view(-1, 1, 32, 32)

    y_pred = model(valid_x)

    loss = loss_funct(y_pred, valid_y)
    iter_loss = loss.item()

    _, predicted = torch.max(y_pred.data, 1)

    iter_acc += torch.sum(predicted == valid_y).item()
    iter_acc /= valid_x.size(0)

    nn_params["validation acc"] = (nn_params['validation acc'] + iter_acc) / 2
    nn_params['validation loss'] = iter_loss

    # ======================= testing data =====================

    iter_acc = 0
    iter_loss = 0

    test_x = test_x.view(-1, 1, 32, 32)

    y_pred = model(test_x)

    loss = loss_funct(y_pred, test_y)
    iter_loss = loss.item()

    _, predicted = torch.max(y_pred.data, 1)

    iter_acc += torch.sum(predicted == test_y).item()
    iter_acc /= test_x.size(0)

    nn_params["testing acc"] = (nn_params['testing acc'] + iter_acc) / 2
    nn_params['testing loss'] = iter_loss

    print(
        f"Iter: {i} | Training Acc: {round(nn_params['training acc']*100, 2)} | Valid Acc: {round(nn_params['validation acc']*100, 2)} | Testing Acc: {round(nn_params['testing acc']*100, 2)}")

    nn_output[0, i] = nn_params['training acc']
    nn_output[1, i] = nn_params['validation acc']
    nn_output[2, i] = nn_params['testing acc']
    nn_output[3, i] = nn_params['training loss']
    nn_output[4, i] = nn_params['validation loss']
    nn_output[5, i] = nn_params['testing loss']

# plotting
x = np.arange(1, max_iters + 1)

plt.figure(1)
plt.title("Q6.1.2 Accuracy Curve")
plt.plot(x, nn_output[0, :] * 100, label="Training Accuracy")
plt.plot(x, nn_output[1, :] * 100, label="Validation Accuracy")
plt.plot(x, nn_output[2, :] * 100, label="Testing Accuracy")
plt.legend()

plt.figure(2)
plt.title("Q6.1.2 Loss Curve")
plt.plot(x, nn_output[3, :], label="Training Loss")
plt.plot(x, nn_output[4, :], label="Validation Loss")
plt.plot(x, nn_output[5, :], label="Testing Loss")
plt.legend()

plt.show()
