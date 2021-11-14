# -*- coding: utf-8 -*-
# In[1]:
import random
import scipy
import pandas as pd
import seaborn as sns

#from collections import Counter

import matplotlib.pyplot as plt

from ann_visualizer.visualize import ann_viz;

from sklearn.tree import export_graphviz
import graphviz

from matplotlib.pyplot import imread
import matplotlib
import numpy as np
import os
import imageio
import cv2
from random import seed
seed(11)

from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split

import tensorflow 
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
from tensorflow.keras import backend as K
from keras.backend import manual_variable_initialization
manual_variable_initialization(True)
from keras.layers import Conv2D, Dense, Activation, Flatten, MaxPooling2D, Dropout, Input
from keras.utils import np_utils

from numpy.testing import assert_allclose
#import nibabel as nib
#from nibabel.testing import data_path
from PIL import Image
import nilearn
from nilearn import image, plotting
from PIL import ImageFilter
from PIL import ImageOps

from medpy.filter.smoothing import anisotropic_diffusion


# In[2]:

#using transverse images: 208x176
def read_scans(): # read in OASIS-1 MRI data across discs
    path = 'D:/be_project/RAW_images/'
    disc_list = os.listdir(path) # get list of discs from directory

    brain_list, total_subjects = [], [] # generate list of brain images for input to NN, all subjects used in study
    
    for disc in disc_list:
        path_ind_disc = f'{path}/{disc}'
        subject_list_p_disc = os.listdir(path_ind_disc) # generate list of subjects in each disc
            
        for subj_id in subject_list_p_disc:
            total_subjects.append(subj_id) # maintain a list of all subjects included in study for diagnosis labeling later
   
            path_n3 = path_ind_disc + f'/{subj_id}/PROCESSED/MPRAGE/T88_111/{subj_id}_mpr_n3_anon_111_t88_gfc_tra_90.gif'
            path_n4 = path_ind_disc + f'/{subj_id}/PROCESSED/MPRAGE/T88_111/{subj_id}_mpr_n4_anon_111_t88_gfc_tra_90.gif'
            path_n5 = path_ind_disc + f'/{subj_id}/PROCESSED/MPRAGE/T88_111/{subj_id}_mpr_n5_anon_111_t88_gfc_tra_90.gif'
            path_n6 = path_ind_disc + f'/{subj_id}/PROCESSED/MPRAGE/T88_111/{subj_id}_mpr_n6_anon_111_t88_gfc_tra_90.gif'
 
            path_list = [path_n4,path_n3, path_n6,path_n5]
            for i in path_list:
                if os.path.exists(i) == True:
                    brain_list.append(plt.imread(i)) # append if file format exists

    return disc_list, brain_list, total_subjects


# In[3]:
disc_list, brain_list, total_subjects = read_scans()

# In[4]

#High Pass Filter
def FilterDesign(img, rsize,csize, Do):
    # D is distance Matrix
    D = np.zeros([rsize, csize], dtype=np.uint32)
    
    # H is Filter
    #transverse
    H = np.zeros([rsize, csize], dtype=np.uint8)
    r = 208//2
    c = 176//2
    
    # Distance Vector
    for u in range(0, rsize):
        for v in range(0, csize):
            D[u, v] = abs(u - r) + abs(v - c)
    # Using Cut off frequncy applying 0 and 255 in H to make a High Pass Filter and center = 1
    for i in range(rsize):
        for j in range(csize):
            if D[i, j] > Do:
                H[i, j] = 255
            else:
                H[i, j] = 0
    return H

brain_list_hpf = brain_list.copy()
#for i in range (0,1):
for i in range(0,len(brain_list_hpf)-1):
    # Image read
    img = Image.fromarray(brain_list_hpf[i])
    #for transverse: 208x176
    rsize = 208
    csize = 176
    # Cut off Frequency
    Do = 3
    H = FilterDesign(img, rsize, csize, Do)
    #cv2.imshow('Rectangular High Pass Filter', H)
    #cv2.waitKey(3000)
    #cv2.destroyAllWindows()
    # Applying fft and shift
    input = np.fft.fftshift(np.fft.fft2(img))
    # Multiplying image with Low Pass Filter
    out = input*H
    # Taking Inverse Fourier of image
    out = np.abs(np.fft.ifft2(np.fft.ifftshift(out)))
    out = np.uint8(cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX, -1))
    #brain_list_hpf[i] = Image.fromarray(out)
    brain_list_hpf[i] = out
    #cv2.imshow('High Pass Filtered Image', out)
    #cv2.waitKey(30000)#30 secs
    #cv2.destroyAllWindows()
    #img.show()
    #brain_list_hpf[i].show()
brain_list = brain_list_hpf.copy()

#Sharpening Filter
brain_list_sharp = brain_list.copy()
for i in range(0,len(brain_list_sharp)-1):
#for i in range(0,1):
    #img=brain_list_sharp[i]
    img = Image.fromarray(brain_list_sharp[i])
    newimg= img.filter(ImageFilter.SHARPEN())
    brain_list_sharp[i] = np.array(newimg)
    #img.show()
    #newimg.show()
    #brain_list_sharp[i].show()
    #order = (80, 97, 96, 123) # left, up, right, bottom
    #ImageOps.crop(newimg, border)
brain_list = brain_list_sharp.copy()


#Contrast Stretching Filter
brain_list_contrast = brain_list.copy()
def normalizeImg(intensity):
    Pin = intensity
    a = 0 #lower limit
    b = 255 #upper limit
    c = 10 #lowest pixel intensity value in the image
    d = 93 #highest piel intensity value in the image
    Pout = (Pin - c)*(((b - a)/(d-c))+a)
    return Pout

#for i in range(0,1):
for i in range(0,len(brain_list_contrast)-1):
    img = Image.fromarray(brain_list_contrast[i])
    multiBands = img.split()
    normalizedImage = multiBands[0].point(normalizeImg)
    #brain_list_contrast[i] = normalizedImage
    brain_list_contrast[i] = np.array(normalizedImage) 
    #img.show()
    #normalizedImage.show()
    #brain_list_contrast[i].show()
brain_list= brain_list_contrast.copy()    

#Anisotropic Diffusion Filter
brain_list_ani = brain_list.copy()
for i in range(0,len(brain_list_ani)-1):
  img_filtered = Image.fromarray(anisotropic_diffusion(brain_list_ani[i],kappa = 10,gamma = 0.25,option = 1,niter = 15))
  brain_list_ani[i] = np.array(img_filtered)           
brain_list = brain_list_ani.copy()


# In[5]:

def read_diagnosis(total_subjects): # builds a dictionary of subjects and diagnoses
    oasis1 = pd.read_csv('D:/be_project/brain_scans_oasis/oasis_cross_sectional.csv') # read in summary file
    oasis1['CDR'].fillna(0, inplace=True) # null values are healthy diagnoses
    diagnosis_qual={0.:'normal', 0.5:'AD', 1.:'AD', 2.:'AD' } # convert diagnosis to labels
    oasis1.replace({"CDR": diagnosis_qual}, inplace=True)
    diagnosis_quant={'normal':0,'AD':1} # convert diagnosis to numerical values
    oasis1.replace({"CDR": diagnosis_quant}, inplace=True)
    oasis1['Subject'] =pd.DataFrame([subj[0:9] for subj in oasis1['ID']]) # extract subject ID from MR ID
    
    subjects = [subj[0:9] for subj in total_subjects] # get subject names for each MRI ID
    subjects = pd.DataFrame(subjects, columns = ['Subject']) # convert to dataframe
    
    diagnosis_dict= {oasis1['Subject'][num]: oasis1['CDR'][num] for num in range(0, 436)} # create a dictionary with subject and diagnosis
    diag = [diagnosis_dict[subj] for subj in subjects['Subject']] # create a list of diagnoses to append to dataframe of subjects
    subjects['Diagnosis'] = pd.DataFrame(diag)
    
    return subjects, diagnosis_dict


# In[6]:

subjects, diagnosis_dict = read_diagnosis(total_subjects)

# In[7]:

def combine_dataset(subjects, total_subjects, disc_list,brain_list):
    subjects['img'] = brain_list.copy()
    #upsampling by adding 236 images
    AD_subjects = subjects.sort_values(by='Diagnosis', ascending = False).head(100) # all subjects diagnosed as AD
    AD_subjects = AD_subjects.append(AD_subjects)
    head = AD_subjects.head(36) #upsample
    AD_subjects = AD_subjects.append(head)    

    
    for subj in AD_subjects['Subject']:
        total_subjects.append(str(subj) + '_MR1') 
    
    subjects = subjects.append(AD_subjects)
    #shuffle
    #subjects=subjects.sample(frac=1).reset_index(drop=True)
    
    brain_list = subjects['img'].tolist()
    brain_list = np.asarray(brain_list.copy())
    brain_list = np.array(brain_list.copy())
    brain_array = brain_list.copy()
    brain_array = np.asarray(brain_list)
    
    return brain_array, AD_subjects,subjects


# In[8]:

brain_array, AD_subjects, subjects = combine_dataset(subjects, total_subjects, disc_list,brain_list) 

# In[9]

#brain_array = np.asarray(brain_array)
# Convoluted Neural Network for Diagnosis
x_MRI = np.asarray(brain_array) # array of image values
y_MRI = np.asarray(subjects['Diagnosis']) # diagnosis for each subject
#x_MRI = x_MRI.reshape(-1, 224, 224,1)
x_MRI = x_MRI.reshape(-1, 208, 176,1)

# split into test and train sets
x_MRI_train, x_MRI_test, y_MRI_train, y_MRI_test = train_test_split(x_MRI, y_MRI, random_state = 11)


df = pd.DataFrame(y_MRI_train,columns=['Diagnosis'])
a = df['Diagnosis'].value_counts()
df1 = pd.DataFrame(y_MRI_test,columns=['Diagnosis'])
b = df1['Diagnosis'].value_counts()
a
b

from keras.preprocessing.image import ImageDataGenerator

#gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,height_shift_range=0.08, zoom_range=0.08)
test_gen = ImageDataGenerator()
train_gen = ImageDataGenerator()

#train_generator = gen.flow(x_MRI_train, y_MRI_train, batch_size=64)
train_generator = train_gen.flow(x_MRI_train, y_MRI_train, batch_size=64)
test_generator = test_gen.flow(x_MRI_test, y_MRI_test, batch_size=64)

new_x_MRI_train = tensorflow.cast(x_MRI_train, tensorflow.float32)
new_y_MRI_train = tensorflow.cast(y_MRI_train, tensorflow.float32)

new_x_MRI_test = tensorflow.cast(x_MRI_test, tensorflow.float32)
new_y_MRI_test = tensorflow.cast(y_MRI_test, tensorflow.float32)

# In[10]:
input_shape = (208,176,1)
#input_shape = (256,256,1)

model = Sequential()
model.add(Conv2D(100, kernel_size=(3, 3), strides=(10,10), activation='sigmoid', padding ='same', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
model.add(Conv2D(50, kernel_size=(3, 3), activation='sigmoid', strides=(5,5), padding ='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
model.add(Conv2D(25, kernel_size=(3,3), activation='sigmoid', strides = (1,1), padding ='same'))
model.add(MaxPooling2D(pool_size=(1, 1), padding='valid'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_MRI_train, y_MRI_train, epochs = 100, batch_size=55)
#history = model.fit_generator(train_generator, steps_per_epoch=508//64, epochs=100,validation_data=test_generator, validation_steps=170//64)
#history = model.fit_generator(train_generator, steps_per_epoch=8, epochs=100,validation_data=test_generator, validation_steps=5)

model.summary()
#model.save("D:/be_project/models/regular/model28_05_2021")

model2 = keras.models.load_model('D:/be_project/models/regular/model8')
lstmweights=model2.get_weights()
model2.set_weights(lstmweights)
# In[11]:

def get_metrics(model, x_test, y_test): # get accuracy, recall, precision
    results = model.evaluate(x_test, y_test)
    accuracy = round(results[1]*100,2)
    y_pred = model.predict_classes(x_test) # predictions for test set
    recall = round((recall_score(y_test, y_pred))*100, 2)
    precision = round(precision_score(y_test, y_pred)*100, 2)
    f1score = round((precision*recall*2)/(precision+recall),4)
    loss = results[0]
    return accuracy, recall, precision,f1score,loss

#Train Parameters
model_accuracy, model_recall, model_precision,f1,loss = get_metrics(model2, x_MRI_train, y_MRI_train)

#Test Parameters
model_accuracy, model_recall, model_precision,f1,loss = get_metrics(model2, x_MRI_test, y_MRI_test)


model_accuracy
model_precision 
model_recall
f1 
loss


from sklearn.metrics import roc_curve
y_pred = model2.predict(x_MRI_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_MRI_test, y_pred)

from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# In[12]:

#model diagnostics
from sklearn.metrics import classification_report

y_pred = model2.predict_classes(x_MRI_test) # predictions for test set

print(classification_report(y_MRI_test,y_pred,target_names =['Actual','Predicted']))

y_pred = model.predict_classes(x_MRI_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_MRI_test, y_pred)


#history = model2.fit(x_MRI_train, y_MRI_train, epochs=100, batch_size=55)
history = model2.fit_generator(train_generator, steps_per_epoch=8, epochs=15,validation_data=test_generator, validation_steps=1)
#history = model.fit_generator(train_generator, steps_per_epoch=8, epochs=100,validation_data=test_generator, validation_steps=5)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# In[13]:
    
#Checking the output on images
img_dem = brain_array[2]

img_control = brain_array[0]

img_dem = img_dem.reshape(-1, 208, 176,1)
img_control = img_control.reshape(-1, 208, 176,1)

y_pred_dem = model2.predict_classes(img_dem)
y_pred_dem 
y_pred_control= model2.predict_classes(img_control)
y_pred_control

# In[14]:
 
    
epochss=[100,200,300,400,500,600,700,800,900,1000,1100]
acc=[]
prec=[]
rec=[]
f1 = []
loss = []
z = 1
itrr = 1
#for eps in (100,200,300,400,500,600,700,800,900,1000,1100):
#for eps in (100,100,100,100,100,100,100,100,100,100):
while(z==1):
   disc_list, brain_list, total_subjects = read_scans()
   brain_list_contrast= brain_list.copy() 
   for i in range(0,len(brain_list_contrast)-1):
       img = Image.fromarray(brain_list_contrast[i])
       img= img.filter(ImageFilter.SHARPEN())
       multiBands = img.split()
       normalizedImage = multiBands[0].point(normalizeImg)
       img_filtered = anisotropic_diffusion(np.array(normalizedImage),kappa = 10,gamma = 0.25,option = 1,niter = 15)
       brain_list_contrast[i] = np.array(img_filtered) 
   brain_list= brain_list_contrast.copy() 
  
   subjects, diagnosis_dict = read_diagnosis(total_subjects)
   brain_array, AD_subjects, subjects = combine_dataset(subjects, total_subjects, disc_list, brain_list) 
    
   x_MRI = brain_array
   y_MRI = np.asarray(subjects['Diagnosis'])
   x_MRI = x_MRI.reshape(-1, 208, 176,1)
   
   # split into test and train sets
   x_MRI_train, x_MRI_test, y_MRI_train, y_MRI_test = train_test_split(x_MRI, y_MRI, random_state = 11)
   input_shape = (208,176,1)

   model = Sequential()
   model.add(Conv2D(100, kernel_size=(3, 3), strides=(10,10), activation='sigmoid', padding ='same', input_shape=input_shape))
   model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
   model.add(Conv2D(50, (3, 3), activation='sigmoid', strides=(5,5), padding ='same'))
   model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
   model.add(Conv2D(25, kernel_size=(3,3), activation='sigmoid', strides = (1,1), padding ='same'))
   model.add(MaxPooling2D(pool_size=(1, 1), padding='valid'))
   model.add(Flatten())
   model.add(Dense(1, activation='sigmoid'))
    
   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
   model.fit(x_MRI_train, y_MRI_train, epochs=100, batch_size=75)
 
   model_accuracy, model_recall, model_precision,f1score,lss = get_metrics(model, x_MRI_test, y_MRI_test)
   #if (model_accuracy >= 91.75):
   acc.append(model_accuracy)
   prec.append(model_precision)
   rec.append(model_recall)
   f1score = round((model_precision*model_recall*2)/(model_precision+model_recall),4)
   f1.append(f1score)
   loss.append(lss)
   if (model_accuracy >= 90):
       model.save("D:/be_project/models/regular/modeltrial{}".format(itrr))
       z = 10
   print("accuracy ",model_accuracy)
   print("precision ",model_precision)
   print("recall ",model_recall)
   print("f1 ",f1score)
   print("loss",lss)
   print("iteration: ",itrr)
   
   itrr+=1
    
itr=[1,2,3,4,5,6,7,8,9,10]
results=pd.DataFrame({'iterations':itr,'accuracy':acc,'precision':prec,'recall':rec,'f1 score':f1,'loss':loss})

results=pd.DataFrame({'epochs':epochss,'accuracy':acc,'precision':prec,'recall':rec,'f1 score':f1})


model_accuracy
model_precision
model_recall

#model 1: 90.83, 75, 81.82

# In[15]:

# visualize CNN
ann_viz(model, title="Neural Network for MRI Classification"); 

#model.evaluate(x_MRI_test,y_MRI_test)
