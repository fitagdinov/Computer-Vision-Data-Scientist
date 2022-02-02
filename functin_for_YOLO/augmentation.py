# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 12:59:48 2022

@author: Robert
"""
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


            
            
        
    