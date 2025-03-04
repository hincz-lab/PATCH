import os
import cv2
import sys
import time
import hdbscan
import numpy as np
import pandas as pd
import tensorflow as tf
import umap.umap_ as umap
import numpy.random as rng
import plotly.express as px
from patchify import patchify
from keras.models import Model
from keras.layers import Layer
from keras import backend as K
import matplotlib.pyplot as plt
from skimage import img_as_float
import sklearn.cluster as cluster
from sklearn.utils import shuffle
from keras.regularizers import l2
from matplotlib.pyplot import imread
from sklearn.metrics import accuracy_score 
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from keras.layers.pooling import MaxPooling2D
from keras.initializers import glorot_uniform
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from keras.layers.core import Lambda, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential, load_model

import random
from scipy import ndimage
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from sklearn.model_selection import train_test_split

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from tensorflow.keras import regularizers, optimizers, models, layers, losses, metrics
from tensorflow.keras.layers import Dense, Activation, Flatten, AveragePooling2D, Dropout, Input, Conv2D, ZeroPadding2D, concatenate

print("imports complete")


#######################DEF FUNCTS##########################


##load data without any changes (high resolution)
def load_data(file_path):
    data = cv2.imread(file_path,cv2.IMREAD_UNCHANGED)
    data = img_as_float(data)*255
    return data

def crop_image(image, rows, cols):
    section = image[rows[0]:rows[1], cols[0]:cols[1]]
    return section

def resize_patches(patch_list):   
    resize_patches = [None]*len(patch_list)
    for i in range(len(patch_list)):
        resize_patches[i] = cv2.resize(patch_list[i],(224, 224))
    new_list = np.asarray(resize_patches, dtype=np.float64)
    return new_list

# divide each channel of the painting to patches then concatenate channels, also get the list of
# corresponding labels (painting id)
def get_patches(data, patch_size, painter_id):
    pc1 = patchify(data[:,:,0], (patch_size , patch_size), step = patch_size)
    pc1 = pc1.reshape(-1, patch_size, patch_size)
    pc2 = patchify(data[:,:,1], (patch_size , patch_size), step = patch_size)
    pc2 = pc2.reshape(-1, patch_size, patch_size)
    pc3 = patchify(data[:,:,2], (patch_size , patch_size), step = patch_size)
    pc3 = pc3.reshape(-1, patch_size, patch_size)
    pc1_reshaped = pc1.reshape(*pc1.shape,1)
    pc2_reshaped = pc2.reshape(*pc2.shape,1)
    pc3_reshaped = pc3.reshape(*pc3.shape,1)
    patches = np.concatenate((pc1_reshaped,pc2_reshaped,pc3_reshaped),axis=3)
    
    labels = []
    def get_label(painter_id, patch_len):
        labels.clear()
        labels.append(painter_id * patch_len)
        return labels

    list_len = np.ones(len(patches))
    y_list = get_label(painter_id, list_len)
    y_list = np.reshape(y_list,(len(patches),1)) 
                        
    return patches, y_list  # use this when shuffle=False
 
def process_pipeline(file_path, patch_size, painter_id):
    data = load_data(file_path)
    patch_list, labels = get_patches(data, patch_size, painter_id)
    resized_patches = resize_patches(patch_list)
    #random otagonal rotation:
    for i in range(resized_patches.shape[0]):
        resized_patches[i]=ndimage.rotate(resized_patches[i],45,reshape=False)
        turns=random.randint(0,7)
        if(turns>0) :
            resized_patches[i]=ndimage.rotate(resized_patches[i],45*turns,reshape=False)
    preprocessed_patches = preprocess_input(resized_patches)
    return preprocessed_patches, labels    
    

print("functs defined")


##############################DO STUFF!!!!#############################
#input .png files for pairwise assignment training. 
#setup below is to analyze the images painted by painter 1
pp1=['./fgp1.png','./fgp1.png','./fgp2.png']    
#
pp2=['./fgp2.png','./fgp3.png','./fgp3.png']
     

for bb in range(len(pp1)):
    #painta='fgp'+str(pp1[bb])+'.png'
    #paintb='fgp'+str(pp2[bb])+'.png'

    painta=pp1[bb]
    paintb=pp2[bb]

    aaa=painta.strip().split('.')
    bbb=paintb.strip().split('.')

    print(aaa[0]+"V"+bbb[0])

    start= time.time()
    for ii in range(26):
        #######################IMPORT AND PROCESS#######################
      
        ##200x200 = 1x1cm
        patch_size = 200
        
        #load patches
        p1_x, p1_y = process_pipeline(painta, patch_size, 0)
        p2_x, p2_y = process_pipeline(paintb, patch_size, 1)
       

        ###############################CHECK LENGTHS##########################


        allthepy = np.concatenate((p1_y, p2_y))
        clarses, countz = np.unique(allthepy, return_counts=True)
        print(clarses, countz)
        smol=min(countz) #the smallest class
        clarsedict={}
        for i in range(len(clarses)):
            clarsedict[clarses[i]]=countz[i]
        for item in clarsedict.keys():
            print(item, clarsedict[item])


        #############CREATE DICTS#################

        paintdict={}
        paintdict[0.]=np.concatenate((p1_x))
        paintdict[1.]=np.concatenate((p2_x))

        for item in clarsedict.keys():
            sizer=clarsedict[item]
            paintdict[item]=np.reshape(paintdict[item], newshape=(sizer,224,224,3))
            print(paintdict[item].shape)

        def patchnumbereq(paintclarse, sm_clarse, paintkey):
            #SAMPLE WITH REPLACEMENT FOR BOOTSTRAPPING
            #paintclarse = 1d array from paintdict
            #sm_clarse = #patches in smallest class
            #paintkey is key from paintdict identifying class

            berg=list(range(0,len(paintclarse)))
            pclarse_xrand=np.full((sm_clarse,224,224,3),0.0)
            pclarse_ysmol=np.full((sm_clarse,1),0.0)

            for j in range(sm_clarse):
                partch = np.random.choice(berg, replace=False)
                pclarse_xrand[j]=paintclarse[partch]

            for k in range(smol):
                pclarse_ysmol[k]=paintkey
            print(len(pclarse_xrand))
            return pclarse_xrand, pclarse_ysmol  
        
        
        def patchwork(paintclarse, sm_clarse, paintkey):
            #this makes a checkerboard a la an autoencoder. replace patchnumbereq below
            #paintclarse = 1d array from paintdict
            #sm_clarse = #patches in smallest class
            #paintkey is key from paintdict identifying class

            berg=list(range(0,len(paintclarse), 2))
            pclarse_xrand=np.full((sm_clarse,224,224,3),0.0)
            pclarse_ysmol=np.full((sm_clarse,1),0.0)


            for j in range(int(sm_clarse/2)):
                partch = np.random.choice(berg, replace=False)
                pclarse_xrand[j]=paintclarse[partch]
            for k in range(int(sm_clarse/2)):
                pclarse_ysmol[k]=paintkey
            print(len(pclarse_xrand))
            return pclarse_xrand, pclarse_ysmol



        pc0_x, pc0_y = patchnumbereq(paintdict[0.], smol, 0.)
        pc1_x, pc1_y = patchnumbereq(paintdict[1.], smol, 1.)

        #pc0_x, pc0_y = patchwork(paintdict[0.], smol, 0.)
        #pc1_x, pc1_y = patchwork(paintdict[1.], smol, 1.)

        ########################## CAT ################
      

        x_train_all=np.concatenate((pc0_x, pc1_x))
        y_train_all=np.concatenate((pc0_y, pc1_y))

        print(len(x_train_all)/smol,len(y_train_all)/smol)

        print("done")


        #################!!!!!RUN THE MODEL!!!!!#############

        #define and train model
        folds = 1
        for fold in range(folds):
            print("Fold:", fold)
            x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.3)

            y_train = to_categorical(y_train, num_classes=None)
            y_val = to_categorical(y_val, num_classes=None)


            baseModel = VGG16(weights='imagenet', include_top=False,input_tensor=Input(shape=(224, 224, 3)))
           
            model = models.Sequential()

            model.add(baseModel)
            model.add(layers.AveragePooling2D(pool_size=(3, 3)))
            model.add(layers.Flatten())
            model.add(layers.Dropout(0.25))
            model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
            model.add(layers.Dropout(0.25))
            model.add(layers.Dense(2, activation="sigmoid"))

            for layer in baseModel.layers[:]:
                layer.trainable = True

            model.compile(optimizer=optimizers.Adam(learning_rate = 0.0001), loss='binary_crossentropy', metrics=['accuracy'])
            filepath="weights.best03"+aaa[1]+bbb[1]+'--'+str(ii)+".hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_weights_only=True, save_freq = 'epoch', save_best_only=True, mode='max')
            callbacks_list = [checkpoint]

            history = model.fit(x_train, y_train, epochs=25, batch_size=32, validation_data=(x_val,y_val),shuffle=True,       callbacks=callbacks_list, verbose=2)
           

        model.save("model_reuse_recycle")

        print("done")
        end=time.time()
        extime = end-start
        print(extime)