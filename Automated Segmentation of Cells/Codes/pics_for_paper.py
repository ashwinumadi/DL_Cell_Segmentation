'''import cv2
import numpy as np
import imutils

img_b = cv2.cvtColor(cv2.imread('../matlab_extracted/train_new_back/23-0.png'),cv2.COLOR_BGR2GRAY)
img_o  =cv2.imread('../matlab_extracted/train_new_img/23-0.png')
cv2.imwrite('cm_23_bw.png',img_b)
cv2.imwrite('cm_23_orig.png',img_o)
cnts_c = cv2.findContours(img_b, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts_c = imutils.grab_contours(cnts_c)
for c in cnts_c:
    cv2.drawContours(img_o,[c],0,(0,0,255),6)
cv2.imwrite('cm_23_col.png',img_o)

img_b = cv2.cvtColor(cv2.imread('../matlab_extracted/train_new_back/25-0.png'),cv2.COLOR_BGR2GRAY)
img_o  =cv2.imread('../matlab_extracted/train_new_img/25-0.png')
cv2.imwrite('cm_25_bw.png',img_b)
cv2.imwrite('cm_25_orig.png',img_o)
cnts_c = cv2.findContours(img_b, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts_c = imutils.grab_contours(cnts_c)
for c in cnts_c:
    cv2.drawContours(img_o,[c],0,(0,0,255),6)
cv2.imwrite('cm_25_col.png',img_o)

img_b = cv2.cvtColor(cv2.imread('../matlab_extracted/test_for_nuc_results/31.png'),cv2.COLOR_BGR2GRAY)
img_o  =cv2.resize(cv2.imread('../matlab_extracted/test_new_img/31-0.png'),(256,256))
cv2.imwrite('nm_31_bw.png',img_b)
cv2.imwrite('nm_31_orig.png',img_o)
cnts_c = cv2.findContours(img_b, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts_c = imutils.grab_contours(cnts_c)
for c in cnts_c:
    cv2.drawContours(img_o,[c],0,(0,255,255),2)
cv2.imwrite('nm_31_col.png',img_o)

img_b = cv2.cvtColor(cv2.imread('../matlab_extracted/test_for_nuc_results/27.png'),cv2.COLOR_BGR2GRAY)
img_o  =cv2.resize(cv2.imread('../matlab_extracted/test_new_img/27-0.png'),(256,256))
cv2.imwrite('nm_27_bw.png',img_b)
cv2.imwrite('nm_27_orig.png',img_o)
cnts_c = cv2.findContours(img_b, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts_c = imutils.grab_contours(cnts_c)
for c in cnts_c:
    cv2.drawContours(img_o,[c],0,(0,255,255),2)
cv2.imwrite('nm_27_col.png',img_o)

import cv2, os
import numpy as np  
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture as GM
import matplotlib.pyplot as plt
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

def show_val(img,val):
    temp = []
    for i in range(len(img)):
        temp.append([])
        for j in range(len(img[i])):
            if(img[i][j] == val):
                temp[i].append(0)
            else:
                temp[i].append(255)
    temp = np.array(temp).astype(np.uint8)
    #kernel = np.ones((5,5),np.uint8)
    #temp = cv2.dilate(temp,kernel,iterations = 1)
    
    return temp
def foo():
    images = 1#next(os.walk('../matlab_extracted/90_new_img'))[2]
    for j in range(images):
        img = cv2.imread('C:/Users/ashwin/Documents/CIS project/Nuclei_segmentation/Full Project/aus_datasets/EDF/EDF003.png')
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(512,512))
        kernel = np.ones((10,10),np.uint8)/100
        img = cv2.filter2D(img,-1,kernel)
        img = cv2.blur(img,(5,5))
        Z = img.reshape((-1,1))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K_n =4
        ret,label,center=cv2.kmeans(Z,K_n,None,criteria,10,cv2.KMEANS_USE_INITIAL_LABELS,centers = np.array([[234,123],[112,112],[123,234]]))
        #print('centter--------------------------------')
        #print(center)
        values = []
        for i in range(K_n):
            values.append(int(center[i][0]))
        #print('label-------------------------------')
        #print(label)
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        kernel = np.ones((10,10),np.uint8)/100
        #closing = cv2.morphologyEx(res2, cv2.MORPH_CLOSE, kernel)
        #closing = cv2.filter2D(res2,-1,kernel)
        closing  = cv2.blur(res2,(5,5))
        #cv2.imshow('cloin',closing)
        #cv2.waitKey(0)
        paste_img = show_val(closing,max(values))
        #cv2.imshow('temp',paste_img)
        #cv2.waitKey(0)
        
        cv2.imwrite('temp1.png',paste_img)
        return cv2.imread('temp1.png')

        #cv2.imshow('disp_one val',show_val(res2,max(values)))
        #cv2.imshow('res2',res2)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
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

img = cv2.resize(cv2.imread('nuclues_result.png'),(256,256))
img_b = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cnts_c = cv2.findContours(img_b, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts_c = imutils.grab_contours(cnts_c)
Test = []
Test_n = []
Test_b = []
Test = [cv2.resize(cv2.imread('C:/Users/ashwin/Documents/CIS project/Nuclei_segmentation/Full Project/aus_datasets/EDF/EDF003.png'),(256,256))]*len(cnts_c)

oo = cv2.resize(cv2.imread('C:/Users/ashwin/Documents/CIS project/Nuclei_segmentation/Full Project/aus_datasets/EDF/EDF003.png'),(256,256))
orig = cv2.resize(cv2.imread('C:/Users/ashwin/Documents/CIS project/Nuclei_segmentation/Full Project/aus_datasets/EDF/EDF003.png'),(256,256))
for c in range(len(cnts_c)):
    temp = np.zeros((256,256),dtype = np.uint8)
    cv2.drawContours(temp,[cnts_c[c]],0,255,-1)
    cv2.drawContours(oo,[cnts_c[c]],0,(255,255,0),-1)
    cv2.imwrite('temp.png',temp)
    Test_n.append(cv2.imread('temp.png'))
cv2.imshow('oo',oo)
cv2.waitKey(0)
Test_b =  [cv2.resize(foo(),(256,256))]*len(cnts_c)
cv2.imshow('oo1',Test_b[1])
cv2.waitKey(0)
model = load_model('../h5 files/unet_final_adam_1.h5', custom_objects={'mean_iou': mean_iou})
preds_test = model.predict([Test,Test_n,Test_b], verbose=1)
preds_test_t = (preds_test > 0.78).astype(np.uint8)
preds_test_t[preds_test_t == 1] = 255


for i in range(len(Test)):
    
    #name = temp_new_img[i]
    #cv2.imwrite('../aus_datasets/90_final_results/'+name, np.squeeze(preds_test_t[i]))
    ##cv2.imwrite('../aus_datasets/90_final_groundtruth/'+name, Test_c[i])
    #print('1 - ',i)
    
    img = np.squeeze(preds_test_t[i])
    img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    #cv2.imshow('i',img)
    #cv2.waitKey(0)
    #cv2.imwrite('./cytos-indi-cyto/cy_c_98_'+str(i)+'.png', img)
    #cv2.imwrite('./cytos-indi-nuc/cy_n_98_'+str(i)+'.png', Test_n[i])
    
        
    cnts_n = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts_n = imutils.grab_contours(cnts_n)
    #cv2.imwrite('cy_98_orig.png',Test[2])
    #cv2.imwrite('cy_98_back.png',Test_b[1])
    cv2.imshow('out',img)
    cv2.imshow('nuc',Test_n[i])
    cv2.waitKey(0)
    for c in cnts_n:
        
        if(i%4 == 0):
            cv2.drawContours(orig, [c],0,(127,69,255),1)
        elif(i%5 == 0):
            cv2.drawContours(orig, [c],0,(255,0,255),1)
        elif(i%6 == 0):
            cv2.drawContours(orig, [c],0,(0,255,255),1)
        elif(i%7 == 0):
            cv2.drawContours(orig, [c],0,(100,40,255),1)
        elif(i%8 == 0):
            cv2.drawContours(orig, [c],0,(180,127,255),1)
        else:
            cv2.drawContours(orig, [c],0,(123,255,98),1)
    
    cv2.imshow('orig',orig)
    cv2.waitKey(0)
    cv2.imwrite('EDF_02.png',orig)

'''    
import os
import cv2
'''
ad = os.getcwd()
files = next(os.walk(ad))[2]
aa = ['27','77','80','88','99']
a = 0
for i in range(len(files)):
    temp=files[i].split('_')
    if(temp[0] == 'updated'):
        cv2.imwrite('resized_updated_'+aa[a]+'.png',cv2.resize(cv2.imread(files[i]),(512,512)))
        a = a+1
'''
cv2.imwrite('resized_updated_27_threshold_0.50.png',cv2.resize(cv2.imread('updated_cy_27_col-t.50.png'),(512,512)))
cv2.imwrite('resized_updated_27_threshold_0.90.png',cv2.resize(cv2.imread('updated_cy_27_col-t95.png'),(512,512)))
