import os
import sys
import random
import warnings
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import imutils
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def foo(a):
    b = a.split('.')
    return int(b[0])

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
seed = 42
random.seed = seed
np.random.seed = seed
a_d = os.getcwd()
'''
x_train1 = next(os.walk('../matlab_extracted/train_for_nuc_i'))[2]

x_train1 = sorted(x_train1,key = foo)
print(x_train1)
X_train1 = []
os.chdir('../matlab_extracted/train_for_nuc_i')
#iii=1
for i in x_train1:
    img = cv2.imread(i)
    img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
    X_train1.append(img)

X_train1 = np.array(X_train1).astype(np.uint8)
os.chdir(a_d)


x_train2 = next(os.walk('../matlab_extracted/train_for_nuc_n'))[2]
x_train2 = sorted(x_train2,key = foo)
X_train2 = []
os.chdir('../matlab_extracted/train_for_nuc_n')

for i in x_train2:
    img = cv2.imread(i)
    img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
    X_train2.append(img)

X_train2 = np.array(X_train2).astype(np.bool)
os.chdir(a_d)
'''
# Build U-Net model
inputsA = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
sA = Lambda(lambda x: x / 255) (inputsA)
s = sA #concatenate([sA,sB,sC])
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')  (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(3, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=inputsA, outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()
'''
# Fit model
earlystopper = EarlyStopping(patience=20, verbose=1)
checkpointer = ModelCheckpoint('../h5 files/unet_final_adam_nucleus.h5', verbose=1, save_best_only=True)

#results = model.fit(X_train1 ,X_train2, validation_split=0.1, batch_size=24, epochs=50, 
#                    callbacks=[earlystopper, checkpointer])

'''

#============================================== test part here ===============================================================================================================================
model = load_model('../h5 files/unet_t1.h5', custom_objects={'mean_iou': mean_iou})
'''x_train3 = next(os.walk('../matlab_extracted/90_temp_img'))[2]
x_train3 = sorted(x_train3,key = foo)
X_train3 = []
os.chdir('../matlab_extracted/90_temp_img')
#iii=1
for i in x_train3:
    img = cv2.imread(i)
    img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
    X_train3.append(img)

X_train3 = np.array(X_train3).astype(np.uint8)'''
X_train3 = [cv2.resize(cv2.imread('C:/Users/ashwin/Documents/CIS project/Nuclei_segmentation/Full Project/aus_datasets/EDF/EDF003.png'),(256,256))]
X_train3 = np.array(X_train3).astype(np.uint8)
os.chdir(a_d)
preds_test = model.predict(X_train3, verbose=1)
preds_test_t = (preds_test > 0.5).astype(np.uint8)
preds_test_t[preds_test_t == 1] = 255

for i in range(1):
    #name = x_train3[i]
    #cv2.imwrite('../matlab_extracted/90_test_for_nuc_results/'+name, np.squeeze(preds_test_t[i]))
    cv2.imwrite('nuclues_result.png', np.squeeze(preds_test_t[i]))
    print('1 - ',i)
'''for i in range(900):
    img = cv2.cvtColor(np.squeeze(preds_test_t[i]),cv2.COLOR_BGR2GRAY)
    orig = cv2.cvtColor(X_train3[i],cv2.COLOR_BGR2GRAY)
    print('img, ',img.shape)
    print('xtrain1, ',X_train3[i].shape)
    #plt.imshow(img)
    #plt.show()
    
    cnts_n = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts_n = imutils.grab_contours(cnts_n)
    for c in cnts_n:
        cv2.drawContours(orig,[c],0,(0,255,0),2)
    cv2.imshow('disp',orig)
    cv2.waitKey(0)'''
