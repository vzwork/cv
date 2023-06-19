from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
from tensorflow.keras.models import Sequential
import numpy as np

from tensorflow.keras.optimizers import Adam

def load_train(path):
    features_train = np.load(path + 'train_features.npy')
    target_train = np.load(path + 'train_target.npy')
    # features_train = features_train.reshape(features_train.shape[0], 28 * 28) / 255.
    features_train = features_train.reshape(features_train.shape[0], 28, 28, 1) / 255
    return features_train, target_train

def create_model(input_shape):
    optimizer = Adam(lr=0.0001)

    model = Sequential()
    model.add(Conv2D(10, 6, padding='same', activation='relu', input_shape=input_shape))
    # model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(10, 5, padding='valid', activation='relu'))
    # model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(10, 4, padding='valid', activation='relu'))
    
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    return model

def train_model(model, train_data, test_data, batch_size=32, epochs=30,
               steps_per_epoch=None, validation_steps=None):

    features_train, target_train = train_data
    features_test, target_test = test_data

    print(target_train.shape[0])

    model.fit(features_train, target_train, 
              validation_data=(features_test, target_test),
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)

    return model 