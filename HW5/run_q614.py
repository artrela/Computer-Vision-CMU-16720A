
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

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.Resize((256, 256))
])


trainset = torchvision.datasets.ImageFolder(
    root='data/SUN_dataset/train', transform=transform)

batch_size = 50

trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, pin_memory=True)

testset = torchvision.datasets.ImageFolder(
    root='data/SUN_dataset/test', transform=transform)
testloader = DataLoader(testset, pin_memory=True)


# establish params
max_iters = 20
# pick a batch size, learning rate
learning_rate = 1e-3
hidden_size = 64

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
    torch.nn.Linear(1745, 8),
    torch.nn.LogSoftmax(dim=1)).to(device)

loss_funct = nn.NLLLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

nn_params = {"training acc": 0,
             "training loss": 0,
             "testing acc": 0,
             "testing loss": 0
             }

nn_output = np.zeros(shape=(4, max_iters))

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

    print(
        f"Iter: {i} | Training Acc: {round(nn_params['training acc']*100, 2)} ")

    nn_output[0, i] = nn_params['training acc']
    nn_output[1, i] = nn_params['training loss']

    # ======================= testing data =====================

    iter_acc = 0
    iter_loss = 0

    for batch_test in testloader:

        test_x, test_y = batch_test

        y_pred = model(test_x)

        loss = loss_funct(y_pred, test_y)
        iter_loss = loss.item()

        _, predicted = torch.max(y_pred.data, 1)

        iter_acc += torch.sum(predicted == test_y).item()

    iter_acc /= len(testset.imgs)

    print("Testing Iteration", i, "Complete.")

    nn_params["testing acc"] = (nn_params['testing acc'] + iter_acc) / 2
    nn_params['testing loss'] = iter_loss

    nn_output[2, i] = nn_params['testing acc']
    nn_output[3, i] = nn_params['testing loss']

    print(
        f"Iter: {i} | Training Acc: {round(nn_params['training acc']*100, 2)} | Testing Acc: {round(nn_params['testing acc']*100, 2)}")

# plotting
x = np.arange(1, max_iters + 1)

plt.figure(1)
plt.title("Q6.1.4 Accuracy Curve")
plt.plot(x, nn_output[0, :] * 100, label="Training Accuracy")
plt.plot(x, nn_output[2, :] * 100, label="Testing Accuracy")
plt.legend()

plt.figure(2)
plt.title("Q6.1.4 Loss Curve")
plt.plot(x, nn_output[1, :], label="Training Loss")
plt.plot(x, nn_output[3, :], label="Testing Loss")
plt.legend()

plt.show()
