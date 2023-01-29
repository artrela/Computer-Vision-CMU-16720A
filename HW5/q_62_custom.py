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
import torchvision
import torchvision.transforms as transforms
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
from torchsummary import summary


batch_size = 128  # 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.Resize((256, 256))
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==================== Load Data =====================

# ? Training Data
trainset = torchvision.datasets.ImageFolder(
    root='../data/oxford-flowers17/train', transform=transform)


trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True)

# ? Testing Data
testset = torchvision.datasets.ImageFolder(
    root='../data/oxford-flowers17/test', transform=transform)
testloader = DataLoader(testset)

# ? Validation Data
valset = torchvision.datasets.ImageFolder(
    root='../data/oxford-flowers17/val', transform=transform)
valloader = DataLoader(valset)

# establish params
max_iters = 10
# pick a batch size, learning rate
learning_rate = 1e-3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Flatten(),
    torch.nn.Linear(59536, 1745),
    torch.nn.ReLU(),
    torch.nn.Linear(1745, 17),
    torch.nn.LogSoftmax(dim=1)).to(device)

loss_funct = nn.CrossEntropyLoss()

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

    for batch_train in trainloader:

        batch_x, batch_y = batch_train

        optimizer.zero_grad()

        y_pred = model(batch_x)

        loss = loss_funct(y_pred, batch_y)
        iter_loss = loss.item()

        loss.backward()

        optimizer.step()

        _, predicted = torch.max(y_pred.data, 1)
        iter_acc += torch.sum(predicted == batch_y).item()

    iter_acc /= len(trainset.imgs)

    nn_params["training acc"] = (nn_params['training acc'] + iter_acc) / 2
    nn_params['training loss'] = iter_loss

    print(f"Training Iteration {i+1} complete")

    # ============================ validation data ===========
    iter_acc = 0
    iter_loss = 0

    for batch_val in valloader:

        batch_x, batch_y = batch_val

        y_pred = model(batch_x)

        loss = loss_funct(y_pred, batch_y)
        iter_loss = loss.item()

        _, predicted = torch.max(y_pred.data, 1)
        iter_acc += torch.sum(predicted == batch_y).item()

    iter_acc /= len(valset.imgs)

    nn_params["validation acc"] = (nn_params['validation acc'] + iter_acc) / 2
    nn_params['validation loss'] = iter_loss

    print(f"Validation Iteration {i+1} complete")

    # ======================= testing data =====================

    iter_acc = 0
    iter_loss = 0

    for batch_test in testloader:

        batch_x, batch_y = batch_test

        y_pred = model(batch_x)

        loss = loss_funct(y_pred, batch_y)
        iter_loss = loss.item()

        _, predicted = torch.max(y_pred.data, 1)
        iter_acc += torch.sum(predicted == batch_y).item()

    iter_acc /= len(testset.imgs)

    nn_params["testing acc"] = (nn_params['testing acc'] + iter_acc) / 2
    nn_params['testing loss'] = iter_loss

    print(f"Test Iteration {i+1} complete")

    print(
        f"Iter: {i+1} | Training Acc: {round(nn_params['training acc']*100, 2)} | Valid Acc: {round(nn_params['validation acc']*100, 2)} | Testing Acc: {round(nn_params['testing acc']*100, 2)}")

    nn_output[0, i] = nn_params['training acc']
    nn_output[1, i] = nn_params['validation acc']
    nn_output[2, i] = nn_params['testing acc']
    nn_output[3, i] = nn_params['training loss']
    nn_output[4, i] = nn_params['validation loss']
    nn_output[5, i] = nn_params['testing loss']

# plotting
x = np.arange(1, max_iters + 1)

plt.figure(1)
plt.title("Q6.2 Custom Accuracy Curve")
plt.plot(x, nn_output[0, :] * 100, label="Training Accuracy")
plt.plot(x, nn_output[1, :] * 100, label="Validation Accuracy")
plt.plot(x, nn_output[2, :] * 100, label="Testing Accuracy")
plt.legend()

plt.figure(2)
plt.title("Q6.2 Custom Loss Curve")
plt.plot(x, nn_output[3, :], label="Training Loss")
plt.plot(x, nn_output[4, :], label="Validation Loss")
plt.plot(x, nn_output[5, :], label="Testing Loss")
plt.legend()

plt.show()
