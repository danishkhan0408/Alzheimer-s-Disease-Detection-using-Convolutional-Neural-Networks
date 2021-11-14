# -*- coding: utf-8 -*-

# In[1]:

import random
import scipy
import pandas as pd
import seaborn as sns

from collections import Counter

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
import keras
from keras import optimizers
from keras.models import Sequential, load_model, Model
from keras.initializers import he_normal
from keras.preprocessing import image
from keras.layers import Conv2D, Dense, Activation, Flatten, MaxPool2D , MaxPooling2D, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras import backend as K
#from tensorflow.contrib.layers import flatten

from numpy.testing import assert_allclose
import nibabel as nib
from nibabel.testing import data_path
from PIL import Image
import nilearn
from nilearn import image, plotting
from PIL import ImageFilter
from PIL import ImageOps
#import ggplot
#from ggplot import aes, geom_point, ggtitle
from medpy.filter.smoothing import anisotropic_diffusion

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import cv2 
# In[2]:

#using transverse images: 208x176
def read_scans(): # read in OASIS-1 MRI data across discs
    path = 'F:/be_project/RAW_images/'
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

# In[4]:

#Anisotropic Diffusion Filter
brain_list_ani = brain_list.copy()
for i in range(0,len(brain_list_ani)-1):
  img_filtered = Image.fromarray(anisotropic_diffusion(brain_list_ani[i],kappa = 10,gamma = 0.25,option = 1,niter = 15))
  brain_list_ani[i] = np.array(img_filtered)           
brain_list = brain_list_ani.copy()

#Sharepening Filter
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


#Increased Contrast Filter
brain_list_contrast = brain_list.copy()
def normalizeImg(intensity):
    Pin = intensity
    a = 0 #lower limit
    b = 255 #upper limit
    #########################values for saggital#####################
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

# In[5]:
def read_diagnosis(total_subjects): # builds a dictionary of subjects and diagnoses
    oasis1 = pd.read_csv('F:/be_project/brain_scans_oasis/oasis_cross_sectional.csv') # read in summary file
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
    #upsampling by adding 242 images
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

# Resizing Images for AlexNet
i = 672
j = 224
k = 224

brain_array_alex = np.int32(np.zeros((i,j,k)))
for i in range(0,len(brain_array)):
    img = brain_array[i]
    img = np.pad(img, [(8, ), (24, )], mode='constant')
    #brain_array_alex[i] = np.int32(np.asarray(img))
    brain_array_alex[i] = np.int32(img)

brain_array = brain_array_alex.copy()

i = 672
j = 224
k = 224
l = 3
brain_array_3d = np.int32(np.zeros((i,j,k,l)))

for x in range(0,len(brain_array)):
    img = brain_array[x]
    r = np.array(img)
    b = np.array(img)
    g = np.array(img) 
    img = cv2.merge((r,b,g))
    brain_array_3d[x] = np.int32(np.asarray(img))


# In[9.1]

x_MRI = np.array(brain_array_3d) # array of image values
y_MRI = np.asarray(subjects['Diagnosis']) # diagnosis for each subject
#x_MRI = x_MRI.reshape(-1, 208, 176,1)
x_MRI = x_MRI.reshape(-1, 224, 224,3)

# split into test and train sets
x_MRI_train, x_MRI_test, y_MRI_train, y_MRI_test = train_test_split(x_MRI, y_MRI, random_state = 11)
#x_MRI_train, x_MRI_test, y_MRI_train, y_MRI_test = train_test_split(x_MRI, y_MRI)
'''
x_MRI_train = x_MRI_train.reshape(x_MRI_train.shape[0], 208, 176, 1)
x_MRI_test = x_MRI_test.reshape(x_MRI_test.shape[0], 208, 176, 1)
'''

x_MRI_train = x_MRI_train.reshape(x_MRI_train.shape[0], 224, 224, 3)
x_MRI_test = x_MRI_test.reshape(x_MRI_test.shape[0], 224, 224, 3)

x_MRI_train = x_MRI_train.astype('float32')
x_MRI_test = x_MRI_test.astype('float32')

x_MRI_train/=255
x_MRI_test/=255

# In[10]:

#SHIVAM VGG16
from keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,height_shift_range=0.08, zoom_range=0.08)

test_gen = ImageDataGenerator()

train_generator = gen.flow(x_MRI_train, y_MRI_train, batch_size=64)
test_generator = test_gen.flow(x_MRI_test, y_MRI_test, batch_size=64)


new_x_MRI_train = tensorflow.cast(x_MRI_train, tensorflow.float32)
new_y_MRI_train = tensorflow.cast(y_MRI_train, tensorflow.float32)

new_x_MRI_test = tensorflow.cast(x_MRI_test, tensorflow.float32)
new_y_MRI_test = tensorflow.cast(y_MRI_test, tensorflow.float32)


train_generator.value_counts()
 # In[11]:
 #VGG
#VGG16
   
from keras.applications.vgg16 import VGG16
#from keras.applications.vgg19 import VGG19
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3)) 
#print(vgg_model.summary())   

from keras.layers import Input, Flatten, Dense, Activation, Dropout
from keras.models import  Sequential

model = Sequential()

model.add(vgg_model)
model.add(Flatten())
model.add(Dense(1))
model.add((Activation('sigmoid')))
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

#model.fit(train_generator, validation_data = test_generator, epochs=60)
#from keras.callbacks import ModelCheckpoint, EarlyStopping
#checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
#history = model.fit_generator(steps_per_epoch=len(x_MRI_train)/100,generator=train_generator, validation_data= test_generator, validation_steps=10,epochs=100,callbacks=[checkpoint,early])
history = model.fit_generator(train_generator, steps_per_epoch=len(x_MRI_train)/100,epochs=15, validation_data=test_generator)

#from keras.callbacks import ModelCheckpoint, EarlyStopping
#checkpoint = ModelCheckpoint("vgg19_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
#model.fit_generator(steps_per_epoch=len(x_MRI_train)/100,generator=train_generator, validation_data= test_generator, validation_steps=10,epochs=300,callbacks=[checkpoint,early])

results = model.evaluate(test_generator)
accuracy = round(results[1]*100,2)
y_pred = model.predict_classes(test_generator) 
recall = round((recall_score(y_MRI_test, y_pred))*100, 2)
precision = round(precision_score(y_MRI_test, y_pred)*100, 2)
f1 = round((precision*2*recall)/(recall+precision),4)
print("accuracy: ",accuracy)
print("precision: ",precision)
print("recall: ",recall)
print("f1 score: ",f1)

'''
from sklearn.metrics import classification_report
target_names = ['Healthy', 'AD']
print(classification_report(y_MRI_test, y_pred, target_names=target_names))
'''

y_pred = model.predict_classes(x_MRI_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_MRI_test, y_pred)

###############################################################################################################################

# In[11]:
   #ALEXNET 
from keras.layers import Input, Flatten, Dense, Activation, Dropout
from keras.models import  Sequential
input_shape = (224,224,3)
image_dim = (224,224,3) # black and white images
n_classes = 2

model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=image_dim, kernel_size=(11,11), strides=(4,4), padding="valid"))
model.add(Activation("relu"))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding="valid"))
model.add(Activation("relu"))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid"))
model.add(Activation("relu"))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid"))
model.add(Activation("relu"))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="valid"))#actual
#model.add(Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding="valid"))
model.add(Activation("relu"))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))#actual
#model.add(MaxPooling2D(pool_size=(1,1), strides=(2,2), padding="valid"))
# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096))
model.add(Activation("relu"))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation("relu"))
# Add Dropout
model.add(Dropout(0.4))

# 3rd Fully Connected Layer
model.add(Dense(1000))
model.add(Activation("relu"))
# Add Dropout
model.add(Dropout(0.4))
n_classes = 2
# Output Layer
model.add(Dense(1))
model.add(Activation("sigmoid"))

# Compile the model
#model.compile(loss="categorical_crossentropy", optimizer="adam", metrics="accuracy")#actual
model.compile(loss="binary_crossentropy", optimizer="adam", metrics="accuracy")



#model.fit(new_x_MRI_train, new_y_MRI_train, epochs=100, batch_size=75)
#model.summary()

#model.fit_generator(train_generator, steps_per_epoch=327//64, epochs=40,validation_data=test_generator, validation_steps=109//64)
history = model.fit_generator(train_generator, steps_per_epoch=508//64, epochs=60,validation_data=test_generator, validation_steps=170//64)


results = model.evaluate(new_x_MRI_test,new_y_MRI_test)
accuracy = round(results[1]*100,2)
y_pred = model.predict_classes(new_x_MRI_test) 
recall = round((recall_score(new_y_MRI_test, y_pred))*100, 2)
precision = round(precision_score(new_y_MRI_test, y_pred)*100, 2)
f1score = round((precision*recall*2)/(precision+recall),4)
loss =  results[0]

print("accuracy: ",accuracy)
print("precision: ",precision)
print("recall: ",recall)
print("f1: ",f1score)
print("loss: ",loss)
y_pred = model.predict_classes(x_MRI_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_MRI_test, y_pred)

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

#from sklearn.metrics import classification_report
#target_names = ['Healthy', 'AD']
#print(classification_report(y_MRI_test, y_pred, target_names=target_names))

#model.save("E:/be_project/python/model/model1")

#model = keras.models.load_model('E:/be_project/python/model/model1')


model.save("E:/be_project/python/model/model1")

model = keras.models.load_model('E:/be_project/python/model/model1')

acc=[]
prec=[]
rec=[]

#for eps in (100,200,300,400,500,600,700,800,900,1000,1100):
for eps in (100,100,100,100,100,100,100,100,100,100):     
    input_shape = (224,224,3)
    image_dim = (224,224,3)
    n_classes = 2
    model = Sequential()
    model.add(Conv2D(filters=96, input_shape=image_dim, kernel_size=(11,11), strides=(4,4), padding="valid"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding="valid"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="valid"))#actual
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))#actual
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation("relu"))
    model.add(Dropout(0.4))
    model.add(Dense(4096))
    model.add(Activation("relu"))
    model.add(Dropout(0.4))
    model.add(Dense(1000))
    model.add(Activation("relu"))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics="accuracy")
    model.fit_generator(train_generator, steps_per_epoch=327//64, epochs=eps,validation_data=test_generator, validation_steps=109//64)
    results = model.evaluate(new_x_MRI_test,new_y_MRI_test)
    accuracy = round(results[1]*100,2)
    y_pred = model.predict_classes(new_x_MRI_test) 
    recall = round((recall_score(new_y_MRI_test, y_pred))*100, 2)
    precision = round(precision_score(new_y_MRI_test, y_pred)*100, 2)
    acc.append(accuracy)
    prec.append(precision)
    rec.append(recall)
    
itr=[1,2,3,4,5,6,7,8,9,10]
results=pd.DataFrame({'iterations':itr,'accuracy':acc,'precision':prec,'recall':rec})

# In[13]:

'''
https://www.kaggle.com/curiousprogrammer/lenet-5-cnn-with-keras-99-48'''

import numpy as np # linear algebra
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(208,176,1)))
model.add(MaxPool2D(strides=2))
model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPool2D(strides=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))
adam = Adam(lr=5e-4)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

# Set a learning rate annealer
reduce_lr = ReduceLROnPlateau(monitor='val_acc', 
                                patience=3, 
                                verbose=1, 
                                factor=0.2, 
                                min_lr=1e-6)

# Data Augmentation
datagen = ImageDataGenerator(
            rotation_range=10, 
            width_shift_range=0.1, 
            height_shift_range=0.1, 
            zoom_range=0.1)
datagen.fit(x_MRI_train)

model.fit_generator(datagen.flow(x_MRI_train, y_MRI_train, batch_size=100), steps_per_epoch=len(x_MRI_train)/100, epochs=30, validation_data=(x_MRI_test, y_MRI_test), callbacks=[reduce_lr])





###################################################################################################################################################################

# In[11]:
def get_metrics(model, x_test, y_test): # get accuracy, recall, precision
    results = model.evaluate(x_test, y_test)
    accuracy = round(results[1]*100,2)
    y_pred = model.predict_classes(x_test) # predictions for test set
    recall = round((recall_score(y_test, y_pred))*100, 2)
    precision = round(precision_score(y_test, y_pred)*100, 2)
    return accuracy, recall, precision

model_accuracy, model_recall, model_precision = get_metrics(model, x_MRI_test, y_MRI_test)

model_accuracy
model_recall
model_precision 

# In[12]:
ann_viz(model, title="Neural Network for MRI Classification"); 
# visualize CNN


