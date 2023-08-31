import os
import cv2
import numpy as np
import imutils
a_d = os.getcwd()


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
def foo1(c):
    val = c.split('-')
    return int(val[0])
def foo2(c):
    val = c.split('.')
    return int(val[0])
def selected_image(check):
    aaa= os.getcwd()
    os.chdir('../matlab_extracted/test_new_nuc')
    nucs = next(os.walk('../matlab_extracted/test_new_nuc'))[2]
    nucs = sorted(nucs, key = foo1)
    ret = []
    for i in nucs:
        ref = i.split('-')
        if(check == ref[0]):
            ret.append(cv2.resize(cv2.cvtColor(cv2.imread(i),cv2.COLOR_BGR2GRAY),(512,512)))
    os.chdir(aaa)
    return ret

def find_the_same(image, results_list, mis_c):
    compare = []
    if(len(results_list)!= 0):
        for i in range(len(results_list)):
            compare.append(foo(results_list[i],image))
        maxi = max(compare)
        ret1 = results_list[compare.index(maxi)]
        temp_arr = np.array(results_list).astype(np.uint8)
        ret2 = np.delete(temp_arr, compare.index(maxi),0)
        ret2 = list(ret2)
        
    else:
        ret1 = np.zeros((512,512),dtype = np.uint8)
        ret2 = results_list
        mis_c = mis_c +1
        print(mis_c)
    return ret1,ret2,mis_c

nuclei = next(os.walk('../matlab_extracted/test_for_nuc_results'))[2]
nuclei = sorted(nuclei, key = foo2)
os.chdir('../matlab_extracted/test_for_nuc_results')
mis_c = 0
for i in range(len(nuclei)):
    img = cv2.resize(cv2.cvtColor(cv2.imread(nuclei[i]),cv2.COLOR_BGR2GRAY),(512,512))    
    cnts_n = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts_n = imutils.grab_contours(cnts_n)
    results_list = []
    for c in range(len(cnts_n)):
        temp = np.zeros((512,512),dtype = np.uint8)        
        cv2.drawContours(temp,[cnts_n[c]],0,255,-1)
        results_list.append(temp)
    len_r_l = len(results_list)
    check = str(i)        
    selected_list=selected_image(check)
    len_s_l = len(selected_list)
    for j in range(len(selected_list)):
        save,results_list,mis_c = find_the_same(selected_list[j],results_list,mis_c)
        
        cv2.imwrite('../matlab_extracted/test_for_nuc_results_arrag/'+check + '-'+str(j)+'.png', np.array(save).astype(np.uint8))
