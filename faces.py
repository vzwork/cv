import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.layers import Dense, Conv2D, Flatten

def load_train(path):
    df_path = path + 'labels.csv'
    folder_path = path + 'final_files/'

    df = pd.read_csv(df_path)
    data_generator = ImageDataGenerator(
        validation_split=0.2,
        scale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90
    )

    # target_size=(128, 128),

    data_flow = data_generator.flow_from_dataframe(
        df,
        folder_path,
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        class_mode='raw',
        subset='training',
        seed=12345
    )
    return data_flow

def load_test(path):
    df_path = path + 'labels.csv'
    folder_path = path + 'final_files/'

    df = pd.read_csv(df_path)
    data_generator = ImageDataGenerator(
        validation_split=0.2,
        scale=1./255
    )

    # target_size=(128, 128),
    # subset='training',

    data_flow = data_generator.flow_from_dataframe(
        df,
        folder_path,
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        class_mode='raw',
        subset='validation',
        seed=12345
    )
    return data_flow

def create_model(input_shape):
    optimizer = Adam(lr=0.001)

    backbone = ResNet50(input_shape=input_shape,
                        weights=None,
                        include_top=False)
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))

    # model = Sequential()
    # model.add(Conv2D(4, 16, activation='relu', padding='valid', input_shape=input_shape))
    # model.add(Conv2D(8, 8, activation='relu', padding='valid'))
    # model.add(Conv2D(8, 8, activation='relu', padding='valid'))
    # model.add(Conv2D(4, 16, activation='relu', padding='same'))
    # model.add(Conv2D(4, 16, activation='relu', padding='same'))
    # model.add(Conv2D(4, 16, activation='relu', padding='same'))
    # model.add(Conv2D(4, 16, activation='relu', padding='same'))
    # model.add(Flatten())
    # model.add(Dense(1, activation='relu'))

    model.compile(optimizer=optimizer, loss='mse',
                  metrics=['mae'])

    return model

def train_model(model, train_data, test_data, 
                batch_size=None, epochs=5,
                steps_per_epoch=None, validation_steps=None):
    
    model.fit(train_data,
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)

    return model