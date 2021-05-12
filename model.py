from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, BatchNormalization

def build_model():

	model = Sequential()
	model.add(Conv2D(input_shape=(48,48,1),filters=32,kernel_size=(3,3),padding="same", activation="relu"))
	model.add(BatchNormalization())
	model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same", activation="relu"))
	model.add(BatchNormalization())
	model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
	model.add(BatchNormalization())
	model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(BatchNormalization())
	model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(BatchNormalization())
	model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
	model.add(BatchNormalization())
	model.add(Flatten())
	model.add(Dense(units=512,activation="relu"))
	model.add(BatchNormalization())
	model.add(Dense(units=512,activation="relu"))
	model.add(BatchNormalization())
	model.add(Dense(units=7, activation="softmax"))

	return model