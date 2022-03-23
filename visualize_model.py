from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.utils.vis_utils import plot_model

model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))#normalize the data
model.add(Conv2D(24, (5, 5), subsample = (2,2)))
model.add(Activation('relu'))
model.add(Conv2D(36, (5, 5), subsample = (2,2)))
model.add(Activation('relu'))
model.add(Conv2D(48, (5, 5), subsample = (2,2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)