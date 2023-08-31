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

def foo(single_nuc, res):
    tot = 0
    num = 0
    for i in range(512):
        for j in range(512):
            if(single_nuc[i][j] == 255):
                tot+=1
            if(single_nuc[i][j] == 255 and res[i][j] == 255):
                num +=1
    acc = num/tot
    return acc
    '''if(acc > .50):
        return 1
    else:
        return 0'''

nuclei = next(os.walk("../matlab_extracted/90_temp_nuc"))[2]
#print(nuclei)

for i in range(len(nuclei)):
    img_n = cv2.imread(os.path.join("../matlab_extracted/90_temp_nuc", str(i+1)+".png")) #i+1
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
        cv2.drawContours(temp_n,[cnts_n[c]],0,255,-1)
    #print(len(cnts_n))
    
    for c in range(len(cnts_n)):
        final = []
        cytos = []
        cytos_iter = next(os.walk('../matlab_extracted/90_temp_cyto'))[2]
        for f in cytos_iter:
            if f.endswith("-"+str(i+1)+".png"): #str(i+1)
                cytos.append(f)
        single_nuc = np.zeros(img_n.shape, dtype = np.uint8)
        
        cv2.drawContours(single_nuc,cnts_n,c,255,-1)
        #cv2.imshow('active',single_nuc)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        final.append(single_nuc)
        #-------------------------------------
        if(cv2.contourArea(cnts_n[c])>1 and cv2.contourArea(cnts_n[c])<1600):
            M = cv2.moments(cnts_n[c])
            if M["m00"] == 0:
                M["m00"] = 0.01
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            center_points = [cY,cX]
        cp_n = center_points    
        #-----------------------------------
        if (c == 0):
            cp_c = []
            cyto_images = []
            for j in range(len(cytos)):
                img_c = cv2.imread("../matlab_extracted/90_temp_cyto/" + cytos[j])
                img_c = cv2.cvtColor(img_c,cv2.COLOR_BGR2GRAY)
                cyto_images.append(img_c)
                cnts_c = cv2.findContours(img_c, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cnts_c = imutils.grab_contours(cnts_c)
                cnts_c_l = len(cnts_c)
                if(cnts_c_l == 1):
                    big_ind = 0
                else:
                    big = 0
                    big_ind = 0
                    for j_i in range(cnts_c_l):
                        area = cv2.contourArea(cnts_c[j_i])
                        if( area >big):
                            big = area
                            big_ind = j_i
                M = cv2.moments(cnts_c[big_ind])
                if M["m00"] == 0:
                    M["m00"] = 0.01
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cp_c.append([cY,cX])
            cyto_images = np.array(cyto_images)
        #print(cp_c)
        consider_cp_c = []
        idx_cp = []
        per = []
        for j in range(len(cyto_images)):
            
            res = cv2.bitwise_and(cyto_images[j], single_nuc)
            #----------------------------------------------------
            
            #----------------------------------------------------
            compare1 = cv2.compare(res,single_nuc,0)
            '''if(compare1.all()):
                
                final[c].append(cyto_images[j])
                print(final[c])'''
            #compare2 = foo(single_nuc,res)
            per.append(foo(single_nuc,res))
            '''if(compare2 == 1):
                final.append(cyto_images[j])
                consider_cp_c.append(cp_c[j])
                idx_cp.append(j)
            '''
            '''
            cv2.imshow('single_nuc',single_nuc)
            cv2.imshow('cytoimagesp[j]',cyto_images[j])
            cv2.imshow('res',res)
            cv2.waitKey(0)    
            cv2.destroyAllWindows()'''
        maxi  =max(per)
        #print(maxi)
        #print(per)
        for j in range(len(per)):
            if(per[j] == maxi):
                final.append(cyto_images[j])
                consider_cp_c.append(cp_c[j])
                idx_cp.append(j)
            
        #print(cyto_images)
        cv2.imwrite('../matlab_extracted/90_new_compare_unet_nuc/'+str(i)+'-'+str(c)+ '.png',final[0])
        #cv2.imshow('nuc',final[0])
        if(len(final) == 2):
            cv2.imwrite('../matlab_extracted/90_new_cyto/'+str(i)+'-'+str(c)+ '.png',final[1])
            #cv2.imshow('cyto',final[1])
            #cv2.waitKey(0)
            cp_c.remove(consider_cp_c[0])
            az = final[1]
            #cyto_images.remove(az.all())
            cyto_images = np.delete(cyto_images, idx_cp[0],0)
            #print(cyto_images)
        else:
            #print(final)
            l = len(final)
            for k in range(l-1):
                current = consider_cp_c[k]
                if k == 0:
                    final_dist = ((cp_n[1]-current[1])**2 + (cp_n[0]-current[0])**2)**1/2
                    final_idx = 0
                else:
                    dist = ((cp_n[1]-current[1])**2 + (cp_n[0]-current[0])**2)**1/2
                    final_dist = min(final_dist,dist)
                    if (final_dist == dist):
                        final_idx = k
            #print(final)
            #print(final_idx)
            cv2.imwrite('../matlab_extracted/90_new_cyto/'+str(i)+'-'+str(c)+ '.png',final[final_idx+1])
            cp_c.remove(consider_cp_c[final_idx])
            cyto_images = np.delete(cyto_images, idx_cp[final_idx],0)
            #cv2.imshow('cyto',final[final_idx+1])
            #cv2.waitKey(0)
        #print(str(i)+'-'+str(c))
        #print('-----------------------------------', c)                

                        
                    
            
