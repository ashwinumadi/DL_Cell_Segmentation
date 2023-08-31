import cv2, os
import numpy as np  
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture as GM
import matplotlib.pyplot as plt

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
    
    return temp

images = next(os.walk('../matlab_extracted/90_new_img'))[2]
for j in images:
    img = cv2.imread('../matlab_extracted/90_new_img/'+j)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(512,512))
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
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(res2, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('cloin',closing)
    #cv2.waitKey(0)
    paste_img = show_val(closing,max(values))
    
    
    
    cv2.imwrite('../matlab_extracted/90_new_back/'+j,paste_img)
    #cv2.imshow('disp_one val',show_val(res2,max(values)))
    #cv2.imshow('res2',res2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    

