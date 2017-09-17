from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential


def create_net(input_shape: tuple, n_classes: int) -> Sequential:
    """
    Define the network architecture and construct it.
    :param input_shape: shape of the input data per one example
    :param n_classes: number of classes to classify
    :return: Sequential model neural network
    """
    model = Sequential()
    model.add(Convolution2D(30, (5, 5), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(60, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    return model


def compile_model(model: Sequential, optimizer='adam', loss='categorical_crossentropy'):
    """
    Create instance of optimization algorithm to train the network
    :param model: neural network to train
    :param optimizer: optimization algorithm implementation
    :param loss: loss function implementation
    """
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

