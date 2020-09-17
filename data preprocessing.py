0# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:24:21 2020

@author: Alex Martin
"""


import matplotlib.pyplot as plt
import sklearn.cluster as sk
import glob
import os
import numpy as np
"""import images for usage of sk.kmeans to make segmentation"""

rgb = plt.imread('celula.jpeg')
rgb2 = plt.imread('celula2.jpeg')
m,n,l = rgb.shape
o,p,q = rgb2.shape
r,l,s = rgb3.shape
rgbr = np.zeros((m,n,6))
rgbr[:,:,0] = rgb[:,:,0]
rgbr[:,:,1] = rgb[:,:,1]
rgbr[:,:,2] = rgb[:,:,2]
rgbr[:,:,3] = rgb2[:,:,0]
rgbr[:,:,4] = rgb2[:,:,1]
rgbr[:,:,5] = rgb2[:,:,2]

vectores = np.reshape(rgbr,(m*n+o*p,3))

kmeans = sk.KMeans(n_clusters=4, random_state=0).fit(vectores)
labels = np.reshape(kmeans.labels_,(240,320,2))

def Segmentacio_100x100(img):
    """finds center of mass form the image and crops the image to a 100x100 pixel rgb image  """
    r,l,n = img.shape
    vector_predict = np.reshape(img,(r*l,3))
    predict = np.reshape(kmeans.predict(vector_predict),(240,320))
    lila = np.uint8(predict==np.ones((240,320))*3)

    x,y = np.meshgrid(np.linspace(0,319,320),np.linspace(0,239,240))
    axis_x = np.int16(np.sum(x*lila)/np.sum(lila))
    axis_y = np.int16(np.sum(y*lila)/np.sum(lila))
    new_img = np.zeros((100,100,3))
    for i in range(0,3):
        imag_padded = np.pad(img[:,:,i], ((50, 50), (50, 50)), 'minimum')
        new_img[:,:,i] = imag_padded[axis_y:axis_y+100,axis_x:axis_x+100]
        
    return new_img/new_img.max()


def files_to_images_saved(folder_dir1,folder_dir2):
    """extracts the image from a folder and applies the segmentation """
    data_path = os.path.join(folder_dir1,'*g')
    src_fname, ext = os.path.splitext(data_path)
    files = glob.glob(data_path)
    train_image = np.zeros((100,100,3), dtype='float32')
    for file in files:
        image = plt.imread(file)
        src_fname, ext = os.path.splitext(file)
        train_image = Segmentacio_100x100(image)
        save_fname = os.path.join(folder_dir2, os.path.basename(src_fname)+'.jpeg')
        plt.imsave(save_fname,train_image)


folders = list(('\EOSINOPHIL','\ZEUTROPHIL','\LYMPHOCYTE','\MONOCYTE'))

for i in folders:
    """INDICAR DIRECCIO DE  LES CARPETES"""
    """img_dir es la d'origen y img_dir_2 es on es guardaran les imatges"""
    img_dir = r"" + i 
    img_dir_2 = r"" + i 
    files_to_images_saved(img_dir,img_dir_2)
    
for i in folders:
    """INDICAR DIRECCIO DE  LES CARPETES"""  
    """img_dir es la d'origen y img_dir_2 es on es guardaran les imatges"""
    img_dir = r"" + i 
    img_dir_2 = r"" + i 
    files_to_images_saved(img_dir,img_dir_2)








