# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 12:51:04 2022

@author: USER
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
            cv.rectangle(new_image,(int(x_lu[i]),int(y_lu[i])),(int(x_rd[i]),int(y_rd[i])),color,thickness) 
    cv.imwrite(directory+'/'+name,new_image)

def augmentation(image,file,directory,augmen,param):
    #image- путь до  изображения
    # file - путь до аннотации (либо txt. либо jpg )
    # directory - папка куда сохранять изображение и аннотацию 
    # augmen- тип аугментации 
    # отражение-"reflect"
    # растяжение-"stretch"
    # сдвиг - "shift"
    # поворот - "rotation"
    #
    # param - параметры аугментации:
    # отражение- 'x' or 'y' параметры оси относительно которой отражение
    # растяжение-(fx,fy) праметры растяжения по x и y
    # сдвиг - (a_x,a_y) параметры сдвига по x и y
    # поворот - alpha угол поворота против часовой стрелки в градусах
    import numpy as np
    import cv2 as cv
    from pathlib import Path
    import math as m
    
    
    name_image=Path(image).name
    name_file=Path(file).name 
    im= cv.imread(image)
    h_im,w_im = im.shape[:2] # size image
    expansion= Path(file).suffix # file expansion( jpg or txt)
    
    if expansion==".txt":
        flag=True # it's jolo.txt
        data=np.loadtxt(file)
        Classes=data[:,0]
        x_mid=data[:,1]
        y_mid=data[:,2]
        width=data[:,3]
        height=data[:,4]
        num=Classes.shape[0] # number of rectagles
    else:
        flag=False # it's mask.jpg
        
    augmen_list=["reflect","stretch","shift","rotat"]
    if augmen not in augmen_list:
        print("Error: you writed wrong augmentation")
        return  
    
    if augmen=="reflect":
        if param=="x": #reflecrion x-axis
            M=np.float32([[1,0,0],[0,-1,h_im]])
        elif param == 'y': #redlection y-axis
            M=np.float32([[-1,0,w_im],[0,1,0]])
        else:
            print("Error: you writed wrong parameter for this augmentation ")
            return 
        new_image=cv.warpAffine(im,M,(w_im,h_im))
        if flag: # jolo.txt
            x_mid=x_mid*M[0,0]+M[0,2]
            y_mid=y_mid*M[1,1]+M[1,2]
            
        else: # mask.jpg
            new_file=cv.warpAffine(file,M,(w_im,h_im))
            
    elif augmen=="stretch":
        if len(param) !=2:
            print("Error: you writed wrong parameter for this augmentation ")
            return
        fx,fy=param
        dim=(int(w_im*fx),(int(h_im*fy)))
        new_image=cv.resize(im,dim,cv.INTER_CUBIC)
        if flag : # mask
            x_mid=np.round(x_mid*fx)
            x_mid=x_mid.astype(np.int32)
            
            y_mid=np.round(y_mid*fy)
            y_mid=y_mid.astype(np.int32)
            
            width= np.round (width*fx)
            width= width.astype(np.int32)
            
            height=np.round(height * fy)
            height=height.astype(np.int32)
            
        else: # jolo.txt
            new_file=cv.resize(file,dim,cv.INTER_CUBIC)
        
    elif augmen=="shift":
        if len(param) !=2:
            print("Error: you writed wrong parameter for this augmentation ")
            return
        a_x,a_y=param
        M=np.float32([[1,0,a_x],[0,1,a_y]])
        new_image=cv.warpAffine(im,M,(w_im,h_im))
        if flag:#jolo.txt
            x_mid=x_mid*M[0,0]+M[0,2]
            y_mid=y_mid*M[1,1]+M[1,2]
        else:#mask
            new_file=cv.warpAffine(file,M,(w_im,h_im))
            
    elif augmen=="rotat":
        alpha=param
        M=cv.getRotationMatrix2D(center=((w_im)/2.0,(h_im/2.0)),angle=alpha,scale=1)
        new_image=cv.warpAffine(im,M,(w_im,h_im))
        if flag:
            x_mid_n=x_mid*M[0,0]+y_mid*M[0,1]+M[0,2]
            y_mid_n=x_mid*M[1,0]+y_mid*M[1,1]+M[1,2]
            x_mid=x_mid_n
            y_mid=y_mid_n
            alpha_rad=alpha*m.pi/180# convert to radians
            width_n=width*abs(m.cos(alpha_rad))+height*abs(m.sin(alpha_rad))
            height_n=height*abs(m.cos(alpha_rad))+width*abs(m.sin(alpha_rad))
            width=width_n
            height=height_n
        else:
            new_file=cv.warpAffine(file,M,(w_im,h_im))
    
    # save file and image
    if flag:
        Classes=np.reshape(Classes,(num,1))
        x_mid=np.reshape(x_mid,(num,1))
        y_mid=np.reshape(y_mid,(num,1))
        width=np.reshape(width,(num,1))
        height=np.reshape(height,(num,1))
        data=np.hstack((Classes,x_mid,y_mid,width,height))
        np.savetxt(directory+'/'+name_file,data)
    else:
        cv.imwrite(directory+'/'+name_file, new_file)
    
    cv.imwrite(directory+'/'+name_image,new_image)    

def random_augmentation(list_photo,list_file,list_augmentation,directory,list_param):
    # list_photo - список из путей фотографий
    # list_file - cписок из путей аннотаций
    #directory- папка для сохранения
    # list_augmentation - список возможных аугментаций
    # list_param - 2х-мерный список для указания диапазона аугментации
    import random
    for i in range(len(list_photo)):
        photo=list_photo[i]
        file=list_file[i]
        index_aug=random.randint(0, len(list_augmentation)-1)#choose random augmentation
        rand_aug=list_augmentation[index_aug]
        range_param=list_param[index_aug]# range parameters for coosed augmentation
        if rand_aug=='reflect':
            param=random.choice(range_param)
        elif rand_aug=="rotat":
            alpha_min, alpha_max=range_param
            param=random.uniform(alpha_min, alpha_max)
        else:# shift or stretch
            min_param , max_param=range_param
            random_x=random.uniform(min_param[0],max_param[0])
            random_y=random.uniform([min_param[1],max_param[1]])
            param=(random_x,random_y)
        augmentation(photo, file, directory, rand_aug, param)
        
        
def xml_to_yolo(File,directory,name_Clsses_file="Classes.json" ):
    # File- путь до файла с рамширением xml "C:\Users\USER\Downloads\ex.xml"
    # directory - папка для сохранения jolo.txt
    # небходимо установить xml
    # name_Clsses_file - имя под которым сохранить словарь с классами в формате json
    import xml.etree.ElementTree as T
    import numpy as np
    from pathlib import Path
    import json
    name_file=Path(File).stem
    tree=T.parse(File)
    root=tree.getroot()
    num= len(root) # number of measurements
    data=np.zeros((num,5))
    Classes={} # list for save Class
    new_index=0 # this used for value new class
    for i in range(num):
        d=list((root[i][2][0][0].attrib.values())) # box info for line
        # format: 'Left', 'Top', 'Right', 'Bottom'
        Class=root[i][2][0].attrib.get("Name") # class for line
        if not(Class in Classes):
            Classes[Class]=new_index
            new_index+=1
        index=Classes[Class]
        d=list(map(int,d)) # convert values in int type
        d=np.array(d)
        x_mid=round((d[0]+d[2])/2)
        y_mid=round((d[1]+d[3])/2)
        width=(d[2]-d[0])
        height=(d[3]-d[1])
        data[i]=np.array([index,x_mid,y_mid,width,height])
    np.savetxt(directory+'/'+name_file+'.txt',data)
    with open(directory+'/'+name_Clsses_file,'w') as file:
        J=json.dumps(Classes)# conver dict in json
        file.write(J)
    return    
