# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:51:36 2020

@author: Alex Martin
"""


"""Code for training the differetns models """
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



def Segmentacio_100x100_phase(img):
    """function that returns the phase of the rgb image """
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

total_length=0
for i in folders:
    """Indicate file path to the processed train images"""
    img_dir = r"" + i 
    train_images_1 , length = files_to_images(img_dir)
    total_length += length 

train_images=np.empty((total_length,100,100,3),dtype='int16')
train_labels=np.empty((total_length),dtype='uint8')
label=0
length_0 = 0
total_length=0

for i in folders:
    """ndicate file path to the processed train images"""
    img_dir = r"" + i 
    train_images_2 , length = files_to_images(img_dir)
    total_length += length 
    train_images[length_0:total_length,:,:,:]=train_images_2
    train_labels[length_0:total_length]=label
    
    label += 1
    length_0 = total_length
    """4 type cell classification use the 2 lines above and for 2 type classification use de 5 lines below"""
    """
    length_0 = total_length
    if i == '\ZEUTROPHIL':
        label += 1
    else:
        label=label
    """
    
total_length=0
for i in folders:
    """ndicate file path to the processed test images"""
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
    """indicate file path to the processed test images"""
    img_dir = r"" + i 
    test_images_2 , length = files_to_images(img_dir)
    total_length += length 
    test_images[length_0:total_length,:,:,:]=test_images_2
    test_labels[length_0:total_length]=label
    length_0 = total_length
    
    label += 1
    length_0 = total_length
    """4 type cell classification use the 2 lines above and for 2 type classification use de 5 lines below"""
    """
    length_0 = total_length
    if i == '\ZEUTROPHIL':
        label += 1
    else:
        label=label
    """

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

"""uncomment the model wanted to use"""

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
