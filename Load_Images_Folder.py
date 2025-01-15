# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 20:44:29 2023

@author: svs26
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets

import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images



#Load Images From Train - Class 1 - NORMAL
train_normal = load_images_from_folder("E:\\Thesis_Cloud_Colab_Scripts\\01_ZhangLab_500\\chest_xray\\train\\NORMAL")


#Load Images From Train - Class 2 - PNEUMONIA
train_pneumonia = load_images_from_folder("E:\\Thesis_Cloud_Colab_Scripts\\01_ZhangLab_500\\chest_xray\\train\\PNEUMONIA")



#Combine Datasets (Train)

#Create Y's for Train dataset



#Load Images From Test - Class 1 - NORMAL
test_normal = load_images_from_folder("E:\\Thesis_Cloud_Colab_Scripts\\01_ZhangLab_500\\chest_xray\\test\\NORMAL")


#Load Images From Test - Class 2 - PNEUMONIA
test_pneumonia = load_images_from_folder("E:\\Thesis_Cloud_Colab_Scripts\\01_ZhangLab_500\\chest_xray\\test\\PNEUMONIA")




#Create a CSV File for the dataset

import csv
import cv2
import os
from pathlib import Path
import shutil


#def Create_Dataloader_Pytorch(folder):
new = "E:\\Thesis_Cloud_Colab_Scripts\\01_ZhangLab_500\\chest_xray\\train_imgdir"

#Path
img_folder = 'E:\\Thesis_Cloud_Colab_Scripts\\01_ZhangLab_500\\chest_xray\\train'

images = []
filenames = []
fl = []
class_counter = 0
for folder in os.listdir(img_folder):
    #Create new subpath
    subfolder = os.path.join(img_folder,folder)
    
    print("extracting filenames from " + str(folder) + "...")
    file = Path(subfolder).glob('*.jpeg')
    for i in file:
        filenames.append(str(i)+","+str(class_counter))
        fl.append(str(os.path.basename(i))+","+str(class_counter))
        shutil.copy(str(i), str(new)+"\\"+str(os.path.basename(i)))
         
        
    class_counter = class_counter+1
    
        

    

#Convert list to pandas Dataframe
import pandas as pd  

df = pd.DataFrame(fl).reset_index(drop=True)
df = df.reset_index(drop=True)
df = df.str.split(",", expand = True)
df.to_csv('zhanglab_train_labels.csv') 
    
    
with open('zhanglab_train_labels.csv', 'w', newline=' ') as csvfile:
    csvtext = csv.writer(csvfile, delimiter=',',
                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
  


csvtext = csv.writer(csvfile, delimiter=',',
                quotechar='|', quoting=csv.QUOTE_MINIMAL)

csvtext.writerow("1") 

csvfile.close();   














#Create a Custom Dataset
import os
import pandas as pd
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
            return len(self.img_labels)
        
        
    def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            image = read_image(img_path)
            label = self.img_labels.iloc[idx, 1]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label
        
    

#Give an "img_dir" image directory for images
img_dir = "E:\\Thesis_Cloud_Colab_Scripts\\01_ZhangLab_500\\chest_xray\\train_imgdir"
csv_file = ""








#Preparing your data for training with DataLoaders
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


#Iterate through the DataLoader
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")








#Train Pytorch Neural Network

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


trainset = torchvision.datasets.CIFAR10(root='dataset', train=True,
                                        download=True, transform=transform)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='dataset', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))



import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 9, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(9, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#net = Net().to(device)  #for GPU
net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        #inputs, labels = data[0].to(device), data[1].to(device) #for GPU
        inputs, labels = data
        
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')


#Save Trained Model

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)



#Test the network on the test data

dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))


import torch

PATH = 'E:/Thesis_Cloud_Colab_Scripts/zhanglab_net.pth'

#Next, let’s load back in our saved model

net = torch.Net()
net.load_state_dict(torch.load(PATH))



#Okay, now let us see what the neural network thinks these examples above are:
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')



# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
