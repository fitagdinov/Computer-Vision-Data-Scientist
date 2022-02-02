# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 10:32:07 2022

@author: Robert
"""
def mask_to_yolo(image,Class,file):
    # image-путь до маски
    # Class- класс объекта 
    #fiel- путь до файла для записи jolo.txt
    import numpy as np
    import cv2 as cv   
    im=cv.imread(image)
    contours, hierarchy = cv.findContours(im,cv.RETR_LIST ,cv.CHAIN_APPROX_SIMPLE)
    array=[]
    for con in contours:
        x,y,w,h = cv.boundingRect(con)
        array.append([Class,int(x+w/2),int(y+h/2),w,h])
    array=np.array(array) 
    np.savetxt(file,array)

