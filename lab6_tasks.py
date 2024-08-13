import numpy as np
import pandas as pd
import datetime
# Codes for generating input data and ground truth labels
# Please DO NOT modify the code
pd.set_option('display.max_columns', None)
dataframe = pd.read_csv('HKProp_Dataset.csv')
dataframe.head()
y = np.array(dataframe['SalePrice_10k'])

# replace '--' with 0, replace $ with space.
dataframe = dataframe.replace('--', '0').replace('$', '')  
Reg_Date = pd.to_datetime(np.array(dataframe['Reg_Date']),infer_datetime_format=True)
Days_to_reg_date = [int(t.days) for t in (Reg_Date - datetime.datetime.today())]
Bedroom = dataframe['Bed_Room'].replace('Studio', '0').fillna(0)
Is_studio = [1 if e == 0 else 0 for e in np.array(dataframe['Bed_Room'])] 

SaleableArea = [int(str(t).replace(',', '').replace('nan', '0')) for t in dataframe['SaleableArea']]
SaleableAreaPrice = [int(str(t).replace(',', '').replace('$', '0').replace('nan', '0')) for t in dataframe['SaleableAreaPrice']]
GrossArea = [int(str(t).replace(',', '').replace('nan', '0')) for t in dataframe['Gross Area']]
GrossAreaPrice = [int(str(t).replace(',', '').replace('$', '0').replace('nan', '0')) for t in dataframe['Gross Area_Price']]

X = np.concatenate([np.array(Days_to_reg_date).reshape(-1, 1), np.array(Bedroom).reshape(-1, 1),
                        np.array(Is_studio).reshape(-1, 1), np.array(SaleableArea).reshape(-1, 1),
                        np.array(GrossAreaPrice).reshape(-1, 1),
                        np.array(GrossArea).reshape(-1, 1),
                        np.array(SaleableAreaPrice).reshape(-1, 1),
                        np.array(pd.get_dummies(dataframe[['Flat', 'Prop_Type', 'Tower', 'Roof']])),
                       np.array(dataframe[['Floor', 'Build_Ages', 'Rehab_Year', 'Kindergarten', 'Primary_Schools', 'Secondary_Schools',
 'Parks', 'Library', 'Bus_Route', 'Mall', 'Wet Market', 'Latitude', 'Longitude']].fillna(0))], axis=1)

X_trainval, X_test, y_trainval, y_test = X[:1000], X[1000:], y[:1000], y[1000:]
print(X_trainval.shape, X_test.shape, y_trainval.shape, y_test.shape)



# DO NOT IMPORT ANY ADDITIONAL LIBRARY! (e.g. sklearn)
def shuffle_data_numpy(X, y, numpy_seed):
    # fix the random seed
    np.random.seed(numpy_seed)

    # TODO Task 1.1
    # shuffle the given data pair (X, y)
    # please use numpy functions so that the results are controled by np.random.seed(numpy_seed)
    concat = np.concatenate((X,y[:, np.newaxis]), axis = 1)
    np.random.shuffle(concat)
    X_shuffle = concat[:, :-1]
    y_shuffle = concat[:, -1]
    

    return X_shuffle, y_shuffle

def train_val_split(X_trainval, y_trainval, train_size, numpy_seed):
    # TODO TASK 1.2 
    # apply shuffle on the data with given random seed, then split the data into training and validation sets
    X_shuffle,y_shuffle = shuffle_data_numpy(X_trainval, y_trainval, numpy_seed)
    X_train = X_shuffle[: train_size , : ]
    y_train = y_shuffle[: train_size ]
    X_val = X_shuffle[train_size : , : ]
    y_val = y_shuffle[train_size : ]

    return X_train, X_val, y_train, y_val
X_train, X_val, y_train, y_val = train_val_split(X_trainval, y_trainval, 700, 42)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
#No additional import allowed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def MyModel(num_dense_layer, dense_layer_unit, input_dim, dropout_ratio):
    # Create a sequential model
    model = Sequential()

    # TODO Task 2.1
    # Build your own model with model.add(), Dense layers, and Dropout layers
    # Hint: you may consider using
        # Dense(): https://keras.io/api/layers/core_layers/dense/
        # Dropout(): https://keras.io/api/layers/regularization_layers/dropout/
    #print(type(input_dim))
    #print(input_dim)
    #model.add(Dense(units = dense_layer_unit, input_shape = (dense_layer_unit, input_dim), activation = 'relu'))
    for i in range (int(num_dense_layer)):
      model.add(Dense(units = dense_layer_unit, input_shape = (input_dim,), activation = 'relu', kernel_initializer="uniform"))
      model.add(Dropout(dropout_ratio, input_shape = (input_dim,)))
    model.add(Dense(units = 1, input_shape = (input_dim,), activation = 'linear',kernel_initializer="uniform" ))
    return model

num_dense_layer = 2
dense_layer_unit = 40
input_dim = len(X_train[0])
dropout_ratio = 0

model = MyModel(num_dense_layer, dense_layer_unit, input_dim, dropout_ratio)
model.summary()

from tensorflow.keras.optimizers import Adam

def MyModel_Training(model, X_train, y_train, X_val, y_val, batchsize, train_epoch):

    # TODO Task 2.2
    # Compile and train the given model
    # Hint: history can be returned by model.fit() function, please see https://keras.io/api/models/model_training_apis/
    adam_optimizer = Adam(learning_rate = 0.001)
    model.compile(optimizer = adam_optimizer, loss = 'mse', metrics = ['mae'])
    history = model.fit(X_train, y_train,
                  epochs=train_epoch,
                  batch_size=batchsize, 
                  validation_data=  (X_val, y_val))
    return history, model

model = MyModel(num_dense_layer, dense_layer_unit, input_dim, dropout_ratio)

batchsize = 4
train_epoch = 150

history, model = MyModel_Training(model, X_train, y_train, X_val, y_val, batchsize, train_epoch)
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
print(f'Test Mean Average Error (MAE): {test_mae}')