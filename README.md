# Basic CNN model using keras sequential API
General framework with setup by step approach for basic CNN model using keras sequential API

* Model Structure
```python
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
model.fit(train, steps_per_epoch=100, validation_data=valid, validation_steps=50, epochs=15,verbose=2)
```
