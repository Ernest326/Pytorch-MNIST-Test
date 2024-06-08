#Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np
import matplotlib.pyplot as plt
import random


#Define the device which we will use
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


#Grab the dataset which we will be using
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)


#Setup data loaders
loaders = {
    'train': DataLoader(
          training_data,
          batch_size=100,
          shuffle=True,
          num_workers=1)
    
    ,'test': DataLoader(
         test_data,
         batch_size=100,
         shuffle=True,
         num_workers=1)
}


#Define the neural network model
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    #Define the activations
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        #Get probabilities for all digits
        return F.softmax(x)


#Display dataset with random sample data
def displayData(training=True):
    ds = training_data if training else test_data

    plt.figure(figsize=(5,5))
    plt.title("MNIST dataset sample data")
    plt.axis("off")

    for i in range(9):
            img_idx = random.randint(1,ds.targets.size()[0])
            ax = plt.subplot(3, 3, i+1)
            plt.imshow(ds.data[img_idx].numpy().astype("uint8"), cmap="gray")
            plt.axis("off")


#Some definitions to use for training/testing
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()


#Train the network
def train(epoch):

    model.train()
    for batch_idx, (data, target) in enumerate(loaders['train']):
          data, target = data.to(device), target.to(device)
          optimizer.zero_grad()
          output = model(data)
          loss = loss_fn(output, target)
          loss.backward()
          optimizer.step()

          if batch_idx % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx*len(data)}/{len(loaders["train"].dataset)} ({100. * batch_idx / len(loaders["train"]):.0f}%)]\t{loss.item():.6f}')


#Test the model with test data
def test():
    
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loaders['test'].dataset)
    print(f'\nAverage loss: {test_loss:.4f}, Accuracy {correct}/{len(loaders["test"].dataset)} ({100. * correct / len(loaders["test"].dataset):.0f}%)\n')


if __name__ == "__main__":

    epochs = 10
    load_model = False
    save_model = True

    #displayData()
    #plt.show()

    if load_model:
        print("Loading model")
        model = torch.load("model")

    print(f"Using {device} device")
    
    for epoch in range(1, epochs):
        train(epoch)
        test()

        if save_model:
            torch.save(model, "model")
            print("Model saved!")