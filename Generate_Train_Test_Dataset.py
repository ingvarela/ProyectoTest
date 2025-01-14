# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 22:44:05 2023

@author: Sergio Varela Santos, M.Sc.

"""

#Import libraries
import os
from pathlib import Path
import shutil
import CustomImageDataset as CID


#Opción para el tipo de encoding de variables
tipo_encoding = "one hot"
#tipo_encoding = "categorical"


#Create Custom Dataset

#def Create_Dataloader_Pytorch(folder):
#new = "E:\\Thesis_Cloud_Colab_Scripts\\01_ZhangLab_500\\chest_xray\\test_imgdir"

#new = "E:\\Thesis_Cloud_Colab_Scripts\\02_Mongomery_500\\test_imgdir"

new = "E:\\Thesis_Cloud_Colab_Scripts\\04_QATAR_300\\Masks\\test_imgdir"

#Path
#img_folder = 'E:\\Thesis_Cloud_Colab_Scripts\\02_Mongomery_500\\test'

#train folder
img_folder = "E:\\Thesis_Cloud_Colab_Scripts\\04_QATAR_300\\Masks\\test"

isExist = os.path.exists(new)
if not isExist:

   # Create a new directory because it does not exist
   os.makedirs(new)
   print(str(new)+" directory is created! ...")

   
images = []
filenames = []
fl=[]


# filenames = "E:\Thesis_Cloud_Colab_Scripts\04_QATAR_300\Masks\train\_Normal\Normal-1.png"

class_counter = 0
for folder in os.listdir(img_folder):
    #Create new subpath
    subfolder = os.path.join(img_folder,folder)
    
    print(subfolder)
    
    print("extracting filenames from " + str(folder) + "...")
    #file = Path(subfolder).glob('*.jpeg')
    file = Path(subfolder).glob('*.png')
    for i in file:
        print()
        filenames.append(str(i)+","+str(class_counter))
        fl.append(str(os.path.basename(i))+","+str(class_counter))
        shutil.copy(str(i), str(new)+"\\"+str(os.path.basename(i)))
            
    class_counter = class_counter+1
    
    

    
#Create Annotations file (fl) into .csv file
filename = 'qatar_mask_test_labels_300.csv'

with open(filename, 'w') as file:
    #Columns Title
    file.write(str("Image Name")+', '+str("Class Label")+'\n')
    #File Name and Class 
    if(tipo_encoding == "one hot"):
        for row in fl:
            rowsplit = row.split(',')
            #Filename
            rowsplit[0]
            #Class Value (for Categorical) 
            
            #use class_counter to determine one hot encoding
            
            
            rowsplit[1]
            #Class Value (for One-Hot Encoding)
            #count the number of classes
            #
            
            file.write(str(rowsplit[0])+', '+str(rowsplit[1])+'\n')
        
        
    else:
        for row in fl:
            rowsplit = row.split(',')
    
    
            #Filename
            rowsplit[0]
            #Class Value (for Categorical) 
            rowsplit[1]
            #Write new row to file
            file.write(str(rowsplit[0])+', '+str(rowsplit[1])+'\n')
        






#Give an "img_dir" image directory for images
img_dir = "E:\\Thesis_Cloud_Colab_Scripts\\04_QATAR_300\\Masks\\train_imgdir"

annotations_file = "E:\\Thesis_Cloud_Colab_Scripts\\qatar_mask_train_labels_300.csv"

        
l = CID.CustomImageDataset(annotations_file, img_dir)

#l = CustomImageDataset(annotations_file, img_dir)
#print(l.img_labels)
#ImageDataset_Labels = l.img_labels



#####################################################################


#Generate the testing set




# #Debug

# import os
# import pandas as pd
# from torchvision.io import read_image
# import numpy as np

# img_labels = pd.read_csv(annotations_file)
# img_dir = img_dir
# #transform = transform
# #target_transform = target_transform

# len(img_labels)


# img_path = os.path.join(img_dir, img_labels.iloc[idx, 0])
# image = read_image(img_path)
# label = img_labels.iloc[idx, 1]

# label = np.zeros((label.size, label.max()+1), dtype=int)
# label[np.arange(label.size),label] = 1 






