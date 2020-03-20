#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np
import os
from random import shuffle
from zipfile import ZipFile

# NOTE: Please download image zip file from box here: https://ibm.box.com/s/5vozosal1gltqjvvn2wjd8r9ctwywlv8

with ZipFile("final_sorted_images.zip", 'r') as zipObj:
    zipObj.extractall()

fire_image_dir = "sorted_images/fire"
normal_image_dir = "sorted_images/selected_normal"

training_data = []

# Label convention [fire, normal]

for image in os.listdir(fire_image_dir):
#     label = [1,0]
    label = 1
    path = os.path.join(fire_image_dir, image)
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    
    if image is not None: 
        image = cv2.resize(image, (224,224))
        training_data.append([np.array(image), label])
    else: 
        pass
    shuffle(training_data)
print("fire images done")

for image in os.listdir(normal_image_dir):
#     label = [0,1]
    label = 0
    path = os.path.join(normal_image_dir, image)
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    
    if image is not None: 
        image = cv2.resize(image, (224,224))
        training_data.append([np.array(image), label])
    else:
        "image_passed"
        pass
    shuffle(training_data)
print("normal images done")

np.save('training_data.npy', training_data)
print("Data saved")
        

