# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 11:58:33 2018

@author: Shalin
"""
#%%
# Import Keras libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#%%
# Initialising the CNN
classifier = Sequential()

#%%
# Convolution
classifier.add(Conv2D(64,(3,3), input_shape = (256, 256, 3), activation = 'relu'))

#%%
# Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#%%
# Add second convolution layer
classifier.add(Conv2D(64,(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Add third convolution layer
classifier.add(Conv2D(128,(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#%%
# Flattening
classifier.add(Flatten())

#%%
# Full Connection
classifier.add(Dense(units = 128, activation = 'relu' ))
classifier.add(Dense(units = 4, activation = 'softmax' ))

#%%
# Compiling
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#%%
# Fitting
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('training_data',
                                                 target_size=(256,256),
                                                 batch_size=16,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('test_data',
                                            target_size=(256, 256),
                                            batch_size=2,
                                            class_mode='categorical')

classifier.fit_generator(training_set,
                    steps_per_epoch=90,
                    epochs=10,
                    validation_data=test_set,
                    validation_steps=4)

#%%
# save classifier
classifier.save('my_model.h5')
#save classifier weights
classifier.save_weights('my_model_weights.h5')