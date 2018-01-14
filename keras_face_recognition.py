# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:38:52 2018

@author: Shalin
"""
#%%
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os

#%%
script_dir = os.path.dirname('D:/new beginning/Project 1')
training_set_path = os.path.join(script_dir, '../training_data')
test_set_path = os.path.join(script_dir, '../test_data')

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(64, (3, 3), input_shape = (256, 256, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#%%
# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('training_data',
                                                 target_size = (128, 128),
                                                 batch_size = 10,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test_data',
                                            target_size = (128, 128),
                                            batch_size = 10,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 65,
                         epochs = 15,
                         validation_data = test_set,
                         validation_steps = 5)
#%%
# Save model
#model_backup_path = os.path.join(script_dir, '/Shalin-model.cpkt')
fname = 'Shalin-model.hdf5'
classifier.save_weights(fname, overwrite = True)
#print("Model saved to", model_backup_path)