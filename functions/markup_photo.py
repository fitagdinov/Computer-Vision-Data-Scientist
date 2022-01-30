# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 12:43:29 2022

@author: Robert
"""
#Функция по наложению Yolo-разметки на фотографию
def markup_photo(markup_path,photo_path,directory,color=(0,255,0),thickness=2):
    #marckup- путь до разметки YOLO
    #photo_path- путь до изображения
    #derectory - папка в которую следует сохранить размеченное изображение
    import numpy as np
    import os
    import cv2 as cv
    photo=cv.imread(photo_path)
    new_image=photo.copy()
    # new_image=cv.resize(new_image,(1024, 768)) # need for scale
    name=os.path.basename(photo_path)
    data=np.loadtxt(markup_path)
    print(data.shape)
    x_mid=data[:,1]
    y_mid=data[:,2]
    width=data[:,3]
    height=data[:,4]
    #find left-up point and right-down point
    x_lu=x_mid - width/2.0
    x_rd=x_mid + width/2.0
    y_lu=y_mid + height/2.0
    y_rd=y_mid - height/2.0 
    for i in range(len(data)):
        if width[i]!=0 and height[i]!=0:
            print(x_lu[i],y_lu[i],x_rd[i],y_rd[i])
            cv.rectangle(new_image,(int(x_lu[i]),int(y_lu[i])),(int(x_rd[i]),int(y_rd[i])),color,thickness) 
    cv.imwrite(directory+'/'+name,new_image)
