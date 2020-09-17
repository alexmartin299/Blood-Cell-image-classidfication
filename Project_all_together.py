# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:51:36 2020

@author: Alex Martin
"""

"""CODI PER AL PROCESAMENT D'IMATGES I GUARDAT A UNA ALTRA CARPETA"""

import matplotlib.pyplot as plt
import sklearn.cluster as sk
import glob
import os
import numpy as np
"""importacio d'imatges per entrenar el model KMeans y poder fer la posterios segmentacio"""

rgb = plt.imread('celula.jpeg')
rgb2 = plt.imread('celula2.jpeg')
m,n,l = rgb.shape
o,p,q = rgb2.shape

rgbr = np.zeros((m,n,6))
rgbr[:,:,0] = rgb[:,:,0]
rgbr[:,:,1] = rgb[:,:,1]
rgbr[:,:,2] = rgb[:,:,2]
rgbr[:,:,3] = rgb2[:,:,0]
rgbr[:,:,4] = rgb2[:,:,1]
rgbr[:,:,5] = rgb2[:,:,2]

vectores = np.reshape(rgbr,(m*n+o*p,3))
"""entrenament del model"""
kmeans = sk.KMeans(n_clusters=4, random_state=0).fit(vectores)
labels = np.reshape(kmeans.labels_,(240,320,2))

def Segmentacio_100x100(img):
    """funcio que troba el centre de massa de la imatge y retalla la imatge a 100x100 pixels 
    amb aquest punt al centre als tres canals RGB y retorna la imatge RGB retallada"""
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
    """extrau les imatges RGB d'una carpeta les segmenta i les torna a guardar amb una altre nom
    a un altre carpeta """
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
"""bucles per realitzar la segmentacio en totes les imatges del dataset"""
for i in folders:
    """INDICAR DIRECCIO DE  LES CARPETES"""
    """img_dir es la d'origen(Train) y img_dir_2 es on es guardaran les imatges(Train_segmented)"""
    img_dir = r"" + i 
    img_dir_2 = r"" + i 
    files_to_images_saved(img_dir,img_dir_2)
    
for i in folders:
    """INDICAR DIRECCIO DE  LES CARPETES"""  
    """img_dir es la d'origen(Test) y img_dir_2 es on es guardaran les imatges(Test_segmented)"""
    img_dir = r"" + i 
    img_dir_2 = r"" + i 
    files_to_images_saved(img_dir,img_dir_2)

"""FINS AQUI EL PROCESSAMENT"""


"""CODI PER A ENTRENAR EL DIFERENTS MODELS"""
"""S'HAN DE FER CAMBIS PER A FER L'ENTRENAMENT DE CADA MODEL """

import numpy as np
import matplotlib.pyplot as plt

import glob
import os
import numpy.fft as fft


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout, SpatialDropout2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

from numpy.random import seed
seed(0)


"""aquesta funcio no s'utilitza cuan el input es RGB"""
def Segmentacio_100x100_phase(img):
    """funcio que rep una imatge rgb i retorna la fase d'aquesta amb les mateixes dimensions """
    r,l,n = img.shape
    
    
    phase_intensity = np.empty((100,100,3),dtype='int16')
    for i in range(0,n):
        fft_1_phase = np.angle((fft.fft2(img[:,:,i]))) 
        fft_1 = 1*(np.cos(fft_1_phase)+np.sin(fft_1_phase)*1j)
        phase = np.real(np.fft.ifft2((fft_1)))
        phase_intensity[:,:,i]= phase/phase.max()*255
        
    # intensity = (0.2989 *new_img[:,:,0] + 0.5870 *new_img[:,:,1] + 0.1140 *new_img[:,:,2])
    # phase_intensity= intensity/intensity.max()
    return phase_intensity


def files_to_images(folder_dir):
    """extrau las imatges rgb d'una carpeta y les retorna en unic array
    a mes de retornar cuantes imatges hi ha a la carpeta"""
    data_path = os.path.join(folder_dir,'*g')
    files = glob.glob(data_path)
    train_images = np.zeros((len(files),100,100,3), dtype='int16')
    i=0
    for file in files:
        image = plt.imread(file)
        train_images[i,:,:,:] = image
        i += 1
    return train_images, len(files)




folders = list(('\EOSINOPHIL','\ZEUTROPHIL','\LYMPHOCYTE','\MONOCYTE'))
"""bucles para extraer las imagenes de las 4 carpetas 
y colocarlas en arrays de train i test y tambien genera los respectivos labels"""
total_length=0
for i in folders:
    """INDICAR DIRECCIO DE  LES CARPETA TRAIN SEGMENTED"""
    img_dir = r"" + i 
    train_images_1 , length = files_to_images(img_dir)
    total_length += length 
"""arrays on es guarden les imatges y es generen les etiquetes per l'entrenament dels models"""
train_images=np.empty((total_length,100,100,3),dtype='int16')
train_labels=np.empty((total_length),dtype='uint8')
label=0
length_0 = 0
total_length=0

for i in folders:
    """INDICAR DIRECCIO DE  LES CARPETA TRAIN SEGMENTED"""
    img_dir = r"" + i 
    train_images_2 , length = files_to_images(img_dir)
    total_length += length 
    train_images[length_0:total_length,:,:,:]=train_images_2
    train_labels[length_0:total_length]=label
    
    label += 1
    length_0 = total_length
    """en cas de voler fer classificacio de 4 tipus de celules utilitzar les dues 
    lines d'adalt en cas de classificacio de 2 tipus utilitzar les 4 linies de sota"""
    """
    length_0 = total_length
    if i == '\ZEUTROPHIL':
        label += 1
    else:
        label=label
    """
    
total_length=0
for i in folders:
    """INDICAR DIRECCIO DE  LES CARPETA TEST SEGMENTED"""
    img_dir = r"" + i 
    test_images_1 , length = files_to_images(img_dir)
    total_length += length 
"""arrays y etiquetes per a la validacio dels models"""
test_images=np.empty((total_length,100,100,3),dtype='int16')
test_labels=np.empty((total_length),dtype='uint8')
label=0
length_0 = 0
total_length=0

for i in folders:
    """INDICAR DIRECCIO DE  LES CARPETA TRAIN SEGMENTED"""
    img_dir = r"" + i 
    test_images_2 , length = files_to_images(img_dir)
    total_length += length 
    test_images[length_0:total_length,:,:,:]=test_images_2
    test_labels[length_0:total_length]=label
    length_0 = total_length
    
    label += 1
    length_0 = total_length
    """en cas de voler fer classificacio de 4 tipus de celules utilitzar les 
    dues lines d'adalt en cas de classificacio de 2 tipus utilitzar les 4 linies de sota"""
    """
    length_0 = total_length
    if i == '\ZEUTROPHIL':
        label += 1
    else:
        label=label
    """
"""significat del numeros a les etiquetes"""
"""labels eosinofilo=0, limfocito=1, monocito =2, neutrofilo=3"""
"""labels (eosinofilo,neutrofilo)polinuclear=0 (monocito,limfocito)mononuclear=1"""

def iteracion(img_n):
    """aplica la funció Segmentacio_100x100_phase a totes les capes dels arrays
    que contenen les imatges d'entrenament y de test"""
    n,r,l,k =img_n.shape
    img_final = np.empty((n,100,100,3),dtype='int16')

    for i in range(0,n):
        img_final[i,:,:,:]=Segmentacio_100x100_phase(img_n[i,:,:,:])

    return img_final

train_images = iteracion(train_images)
test_images= iteracion(test_images)



# train_images , test_images,train_labels,test_labels = train_test_split(train_images, train_labels,test_size=0.25,random_state=42, shuffle=True)

"normalitzacio per a l'entrenament"
train_images  = train_images/ 255.0
test_images = test_images/255.0

"""funcio del modul Tensorflow que genera petites variacion en les imatges per a que el model entreni millor """ 
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

"""callback per a que guardi el model entrenat amb millor validation accuracy"""
my_callbacks = [ModelCheckpoint(filepath='batch_phase_tanh_2types.h5' ,save_best_only=True,monitor='val_accuracy')]

"""DESCOMENTAR EL MODEL QUE ES VULGUI UTILITZAR"""

"""MODEL  CNN de Zhimin Gao et al. """
"""
model = Sequential()

model.add(Conv2D(6, kernel_size=(9,9),
                 activation='tanh',
                 input_shape=(100,100,3)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2),strides=None,))

model.add(Conv2D(16, kernel_size=(5,5),
                 activation='tanh'))
model.add(BatchNormalization())
model.add(MaxPool2D((3,3),strides=None,))          


model.add(Conv2D(32, kernel_size=(3,3),
                 activation='tanh'))
model.add(BatchNormalization())
model.add(MaxPool2D((3,3),strides=None,))


model.add(Flatten())
model.add(Dense(128, activation='tanh'))
model.add(Dense(2, activation='softmax'))
               """                    
"""MODEL  CNN de Zhimin Gao et al. amb relu"""
"""
model = Sequential()

model.add(Conv2D(6, kernel_size=(9,9),
                 activation='relu',
                 input_shape=(100,100,3)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2),strides=None,))

model.add(Conv2D(16, kernel_size=(5,5),
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((3,3),strides=None,))          


model.add(Conv2D(32, kernel_size=(3,3),
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((3,3),strides=None,))


model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))
"""

"""MODEL CNN de Paul Mooney"""
"""
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                  activation='tanh',
                  input_shape=(100,100,3),strides=1))
model.add(Conv2D(64, (3, 3), activation='tanh'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

"""

"""MODEL  CNN de Paul Mooney amb tangent hiperbolica"""
"""
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                  activation='tanh',
                  input_shape=(100,100,3),strides=1))
model.add(Conv2D(64, (3, 3), activation='tanh'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

"""

""" MODEL  combinat"""
"""
model = Sequential()

model.add(Conv2D(6, kernel_size=(9,9),
                  activation='relu',
                  input_shape=(100, 100,3)))

model.add(MaxPool2D((2,2),strides=None,))

model.add(Conv2D(16, kernel_size=(5,5),
                  activation='relu'))

model.add(MaxPool2D((2,2),strides=None))



model.add(Conv2D(32, kernel_size=(3, 3),
                  activation='relu'))

model.add(MaxPool2D((2,2),strides=None,))
model.add(SpatialDropout2D(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
"""

'''compilacio y entrenament utilitzat a tots el models '''
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""entrenament del model de 10 epochs amb batch_size de 20"""
model.fit(datagen.flow(train_images,train_labels, batch_size=20),epochs=10 ,validation_data = [test_images, test_labels],shuffle=True,callbacks=my_callbacks)