# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:16:17 2023

@author: Sergio Varela-Santos, M.Sc.
"""

#Full Run de los experimentos con CNN, para 2 clases (problema binario)
#Cambiar
#img_dir  (linea 39)
#annotations_file  (Linea 40)

# (Linea 110)
# (Linea 111)
# (Linea 288 Class Names)
# (Linea 364 Class Names)



#Number of experiments
for cycle in range(1,30):


    print("running experiment " + str(cycle) + " ...")

    import numpy as np
    import CustomImageDataset as CID
    
    
    #Give an "img_dir" image directory for images from (from E:\Thesis_Cloud_Scripts)
    img_dir = "01_Montgomery_300\\No Mask\\train_imgdir"
    annotations_file = "montgomery_train_labels.csv"
    l = CID.CustomImageDataset(annotations_file, img_dir)
    
    
    
    #Preparing your data for training with DataLoaders
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    
    train_dataloader = DataLoader(l, batch_size=1, shuffle=True)
    #test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    
    
    #Iterate through the DataLoader
    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}") 
    print(f"Labels batch shape: {train_labels.size()}")
    
    # for batch size >1
    # img = train_features[2].squeeze()
    # label = train_labels[2]
    
    
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # print(f"Label: {label}")
    
    # #to verify the type pf tensor exptected
    # print(img.type())
    
    
    
    from torch.nn.modules.activation import Sigmoid
    import time
    import torch.nn as nn
    import torch.nn.functional as F
    
    # PyTorch libraries and modules
    import torch
    from torch.autograd import Variable
    from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Sigmoid, LogSoftmax, NLLLoss
    from torch.optim import Adam, SGD
    
    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # self.conv1 = nn.Conv2d(1, 9, 5)
            # self.pool = nn.MaxPool2d(2, 2)
            # self.conv2 = nn.Conv2d(9, 16, 5)
            # self.fc1 = nn.Linear(16 * 5 * 5, 120)
            # self.fc2 = nn.Linear(120, 84)
            # self.fc3 = nn.Linear(84, 10)
            
            self.cnn_layers = Sequential(
                # Defining a 2D convolution layer
                Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                BatchNorm2d(8),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=3, stride=1),
                # Defining another 2D convolution layer
                Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                BatchNorm2d(16),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=3, stride=1),
                
            )
    
            self.linear_layers = Sequential(
                #Linear(2*150*75, 100),
                Linear(16*296*296, 10),
                Linear(10, 2)
            )
    
            self.output_layers = Sequential(
                #Softmax()
                
                #Binary Classification
                LogSoftmax()
                #Sigmoid()
                
                
            )
            
        # Defining the forward pass    
        def forward(self, x):
            x = self.cnn_layers(x)
            #print(x)
            x = x.view(x.size(0), -1)
            x = self.linear_layers(x)
            x = self.output_layers(x)
            #x = self.NLLLoss(x)
            #print actual tensor
            #print(x)
            return x
    
    
                                        
        # def forward(self, x):
        #     x = self.pool(F.relu(self.conv1(x)))
        #     x = self.pool(F.relu(self.conv2(x)))
        #     x = torch.flatten(x, 1) # flatten all dimensions except batch
        #     x = F.relu(self.fc1(x))
        #     x = F.relu(self.fc2(x))
        #     x = self.fc3(x)
        #     return x
    
    
    #net = Net().to(device)  #for GPU
    net = Net()
    
    
    #trick to obtain image size  --- IMPORTANT
    # input = torch.randn(4, 1, 500, 500)
    # out = net(input)
    # print(out)
    
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)
    
    
    
    
    import torch.optim as optim
    
    #criterion = nn.CrossEntropyLoss()
    
    #For binary problems
    criterion = nn.NLLLoss()
    
    #criterion = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    
    #for a full epoch is the minibatch size * epoch intended = real epochs
    #100 full epochs * 4 minibatch size = 400 epochs
    
    for epoch in range(100):  # loop over the dataset multiple times
    
        print("training epoch..."+str(epoch))
    
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            #inputs, labels = data[0].to(device), data[1].to(device) #for GPU
            inputs, labels = data[0].float(), data[1]
            
            #obtain the desired type for data[1]
            #tipo = data[1].type
            
            #print(tipo)
            
            
            
            
            #convert to one hot encoding
            #labels = F.one_hot(labels, num_classes = 2)
            
            #print(labels)
            
            
            
            
            
            
            
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            #print("training sample..."+str(i))
    
            # print statistics
            running_loss += loss.item()
            #if i % 2000 == 1999:    # print every 2000 mini-batches
            #    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            #    running_loss = 0.0
    
    print('Finished Training')
    
    
    #Save Trained Model
    
    PATH = 'montgomery_300_net_'+str(cycle)+'.pth'
    torch.save(net.state_dict(), PATH)
    
    
    t = time.localtime()
    finish_time = time.strftime("%H:%M:%S", t)
    print("Time of finish: "+str(finish_time))
    
    ############################################################################################################
    
    #Next, let’s load back in our saved model
    #PATH = 'Test_model.pth'
    
    import Net
    import torch
    
    net = Net.Net()
    net.load_state_dict(torch.load('montgomery_300_net_'+str(cycle)+'.pth'))
    net.eval()
  
    
    import CustomImageDataset as CID
    
    #Give an "img_dir" image directory for images
    img_dir = "01_Montgomery_300\\No Mask\\test_imgdir"
    annotations_file = "montgomery_test_labels.csv"
    t = CID.CustomImageDataset(annotations_file, img_dir)
    
    
    
    from torchmetrics.classification import BinaryConfusionMatrix
    ConfMatrix = BinaryConfusionMatrix()
    
    
    #To save data for confusion matrix
    predictions_raw = []
    labels_raw = []
    outputs_list = []
    predictions_list = []
    
    
    #Preparing your data for testing with DataLoaders
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    
    test_dataloader = DataLoader(t, batch_size=1, shuffle=False)
    
    #Iterate through the DataLoader
    # Display image and label.
    test_features, test_labels = next(iter(test_dataloader))
    print(f"Feature batch shape: {test_features.size()}") 
    print(f"Labels batch shape: {test_labels.size()}")
    img = test_features[0].squeeze()
    label = test_labels[0]
    #plt.imshow(img, cmap="gray")
    #plt.show()
    print(f"Label: {label}")
    
    #to verify the type pf tensor exptected
    print(img.type())
    
    
    
    #testing the network
    classes = ('TUBERCULOSIS', 'NORMAL')
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    
    # again no gradients needed
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data[0].float(), data[1]
            outputs = net(images)
            
            _, predictions = torch.max(outputs, 1)
            
            #Obtain predictions per sample
            predictions_raw.append((torch.exp(outputs)))
            
            outputs_list.append(outputs)
            
            predictions_list.append(predictions)
            
            
            
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
                
                #Obtain labels per sample
                labels_raw.append(label)
    
      





          
    #corr_counter = 0
    
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        
        #Add the correct samples to the total
        #corr_counter = correct_count + corr_counter
        
        
        
        
        
    #first parameter is for output tensor index, and second and third are for values inside tensor to display.
    #predictions_raw[4].view(1,4)
    
    
    #print general accuracy and sensitivity, specificity, f1 score.
    
    
    #General Acuracy
    totalsamples = []
    for i in total_pred.values():
        totalsamples.append(i)
        
    correctsamples = []
    for i in correct_pred.values():
        correctsamples.append(i)
        
    Model_Accuracy = (sum(correctsamples)) / (sum(totalsamples))
    
    # Print the sum of the numbers
    print("Accuracy of Testing Set:", Model_Accuracy)
    
    
    
    import torchmetrics
    
    #sensitivity and specificity are for binary classifiers
    #for multiclass use one vs. all apprach
    
    #Stack
    predictions_list = torch.tensor(predictions_list)
    labels_raw = torch.tensor(labels_raw)
    
    #Unsqueeze
    #outputs_list = torch.unsqueeze(outputs_list, dim=0)
    #labels_raw = torch.unsqueeze(labels_raw, dim=0)
    
    
    #Cat
    # outputs_list = torch.cat(outputs_list)
    # labels_raw = torch.cat(labels_raw)
    
    
    # #Sensitivity
    sensitivity = []
    sensitivity = torchmetrics.classification.BinaryRecall()
    Sense = sensitivity(predictions_list,labels_raw)
    

    #Specificity
    specificity = []
    specificity = torchmetrics.classification.BinarySpecificity()
    Speci = specificity(predictions_list,labels_raw)

    
    
    #F1 Score
    f1Score = []
    f1Score = torchmetrics.classification.BinaryF1Score()
    F1S = f1Score(predictions_list,labels_raw)
    #print(F1S)
    
    
    #AUC
    AUC = []
    AUC = torchmetrics.classification.BinaryAUROC()
    Auc = AUC(predictions_list,labels_raw)
    #print(Auc)
    
    
    #Save evaluation metrics in file
    import csv
    
    with open('model_eval_metrics_'+str(cycle)+'.csv', 'w') as csvfile:
        evalwriter = csv.writer(csvfile)
        evalwriter.writerow(['Accuracy (on test set):'])
        evalwriter.writerow([Model_Accuracy])
        evalwriter.writerow(['Sensistivity or Recall:'])
        evalwriter.writerow([Sense.numpy()])
        evalwriter.writerow(['Specificity:'])
        evalwriter.writerow([Speci.numpy()])
        evalwriter.writerow(['F1-Score:'])
        evalwriter.writerow([F1S.numpy()])
        evalwriter.writerow(['Area Under the Curve (AUC):'])
        evalwriter.writerow([Auc.numpy()])


    
    #TUBERCULOSIS,NORMAL
    #Save outputs in one hot encoded
    with open('Output_Matrix_'+str(cycle)+'.csv', 'w') as outmatfile:
        matwriter = csv.writer(outmatfile, delimiter=',')
        #matwriter.writerow(predictions_raw)
        # run a loop for each item of the list
        for Predic in predictions_raw:
            matwriter.writerow(Predic.numpy()[-1])
            matwriter.writerow(Predic.numpy()[0])
        
    
    
    
    