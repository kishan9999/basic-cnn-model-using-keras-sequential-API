
import tensorflow
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import random

# Reproducibility
seed1=19
tensorflow.random.set_seed(seed1)
np.random.seed(seed1)
random.seed(seed1)

# Data Pre-Processing
path1='./datasets/train/'
path2='./datasets/valid/'
train=ImageDataGenerator(rescale=1.0/255.0,
                         rotation_range=2,
                         fill_mode="nearest",
                         height_shift_range=1.01,
                         shear_range=0.01,
                         zoom_range=[1,1.01],
                         horizontal_flip=False).flow_from_directory(path1, 
                                                                    color_mode="grayscale",
                                                                    target_size=(64,64),
                                                                    batch_size=48,
                                                                    shuffle=True)
                             
valid=ImageDataGenerator(rescale=1.0/255.0).flow_from_directory(path2, 
                                                                color_mode="grayscale",
                                                                target_size=(64,64),
                                                                batch_size=24,
                                                                shuffle=True)


# Model Setup
model = Sequential()
model.add(Conv2D(input_shape=(64,64,1),filters=31,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Dropout(0.4))
model.add(Conv2D(filters=16,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=16,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=32,kernel_size=(4,4),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(5,5),padding="same", activation="relu"))
model.add(Conv2D(filters=8,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(units=77,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=6, activation="softmax"))
model.compile(Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

# Visualize
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='my_model0123.png', show_shapes=True, show_layer_names=True)

# Training
hist=model.fit(train, steps_per_epoch=100, validation_data=valid, validation_steps=50, epochs=15,verbose=2)

# Results
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('history: the CNN model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('history: the CNN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()