import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical


class Dataset:
    def __init__(self, subtract_mean=False, normalize = True, shuffle=True):
        (self.train_x, self.train_y), (self.test_x, self.test_y) = mnist.load_data()
        self.n_classes = np.unique(np.concatenate((self.train_y, self.test_y), 0)).size
        if normalize:
            self._normalize()

        if shuffle:
            self._shuffle()

        if subtract_mean:
            self._subtract_mean()

        self._to_categorical()
        self.train_x = self._reshape_data(self.train_x)
        self.test_x = self._reshape_data(self.test_x)

    def num_classes(self):
        """
        :return: number of classes in the dataset
        """
        return self.n_classes

    def example_input_shape(self):
        """
        :return: shape of each example to be fed into the neural network
        """
        return self.train_x.shape[1:]

    def get_training_data(self):
        return self.train_x, self.train_y

    def get_testing_data(self):
        return self.test_x, self.test_y

    def _shuffle(self):
        """
        Shuffle training examples and labels.
        """
        perm = np.random.permutation(self.train_y.shape[0])
        self.train_y = self.train_y[perm]
        self.train_x = self.train_x[perm]
        pass

    def _subtract_mean(self):
        """
        Normalize training and testing examples by subtracting mean training image
        """
        mean_image = np.mean(self.train_x)
        self.train_x -= mean_image
        self.test_x -= mean_image

    def _to_categorical(self):
        """
        Transform each label to binary vector with length the number of classes
        """
        self.train_y = to_categorical(self.train_y, self.n_classes)
        self.test_y = to_categorical(self.test_y, self.n_classes)

    def _reshape_data(self, data: np.ndarray):
        """
        Reshapes the data to fit the backend framework (tensoflow),
        which is (n_images, rows, cols, 1)
        :param data: examples to reshape
        :return reshaped examples
        """
        n_images, rows, cols = data.shape
        return data.reshape(n_images, rows, cols, 1)

    def _normalize(self):
        """
        Transform input data points to matrices of values between 0 and 1
        """
        self.train_x = self.train_x.astype('float32') / 255
        self.test_x = self.test_x.astype('float32') / 255

