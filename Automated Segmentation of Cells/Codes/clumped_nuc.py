# for proper arrangment of cytoplasm in the matlab extracted folder according to the nuclies
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
# construct the argument parse and parse the arguments
from PIL import Image, ImageFilter, ImageEnhance 
import os
import glob

def foo1(c):
    val = c.split('-')
    return int(val[0])
nuclei = next(os.walk("../matlab_extracted/test_new_nuc"))[2]
nuclei = sorted(nuclei, key = foo1)

for i in range(900):
    check = str(i)
    combine = []
    for j in nuclei:
        check_1 = j.split('-')
        if(check == check_1[0]):
            combine.append(j)
    save = np.zeros((512,512),dtype = np.uint8)
    for j in combine:
        temp = cv2.cvtColor(cv2.imread('../matlab_extracted/test_new_nuc/'+j),cv2.COLOR_BGR2GRAY)
        save = np.maximum(save, temp)
    cv2.imwrite('../matlab_extracted/test_for_nuc_n/'+str(i)+'.png', save)
for i in range(900):
    cv2.imwrite('../matlab_extracted/test_for_nuc_i/'+str(i)+'.png', cv2.imread('../matlab_extracted/test_new_img/'+str(i)+'-0.png'))
