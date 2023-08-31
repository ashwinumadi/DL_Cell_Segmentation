import os

orig = next(os.walk('../matlab_extracted/90_new_img'))[2]
dup = next(os.walk('../matlab_extracted/90_new_cyto'))[2]
for i in range(len(orig)):
    if orig[i] not in dup:
        print(orig[i])
'''
import imutils
import numpy as np
import cv2


nuclei = ['20.png']#,'22.png','56.png','65.png','76.png']#next(os.walk("../matlab_extracted/extra_temp_nuc"))[2]
#print(nuclei)

for i in range(len(nuclei)):
    img_n = cv2.imread(os.path.join("../matlab_extracted/extra_temp_nuc", str(22)+".png")) #i+1
    img_n = cv2.cvtColor(img_n,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('main',img_n)
    #cv2.waitKey(0)
    temp_n = np.zeros((512,512),dtype = np.uint8)
    kernel = np.ones((7,7),np.uint8)
    img_n = cv2.dilate(img_n,kernel,iterations = 1)
    cnts_n = cv2.findContours(img_n, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts_n = imutils.grab_contours(cnts_n)
    
    for c in range(len(cnts_n)):
        cv2.drawContours(temp_n,[cnts_n[c]],0,(255,0,0),1)
        #cv2.imshow('imgn',temp_n)
        #cv2.waitKey(0)
    img_n = temp_n
    
    cnts_n = cv2.findContours(img_n, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts_n = imutils.grab_contours(cnts_n)
    for c in range(len(cnts_n)):
        temp_n = np.zeros((512,512),dtype = np.uint8)
        cv2.drawContours(temp_n,[cnts_n[c]],0,255,-1)
        cv2.imshow('see',temp_n)
        cv2.waitKey(0)
    print('----')'''
