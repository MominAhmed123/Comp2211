from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Flatten


def get_datagen() -> ImageDataGenerator:
  datagen = None
  ###############################################################################
 # TODO: your code starts here
 datagen = ImageDataGenerator(horizontal_flip = True, rescale = 1./16, shear_range = 0.2)
  # TODO: your code ends here
  
  return datagen

def custom_model():
  model = None
  ###############################################################################
  # TODO: your code starts here
  model = Sequential()
  model.add(Conv2D(filters = 512, kernel_size = (3,3), activation = 'relu', strides = (2,2), input_shape = (64,64,3)))
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(Conv2D(filters = 400, kernel_size = (3,3), activation = 'relu'))
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(Dropout(0.1))
  model.add(Flatten())
  model.add(Dense(units =512, activation = 'relu'))
  model.add(Dense(units = 15, activation = 'softmax'))





  # TODO: your code ends here
  ###############################################################################
  return model
