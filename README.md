# MNIST_conv
Learning a shallow convolutional network example on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

# Installation
To install go to digits folder and type `pip install -r requirements`

# Run training and evaluation
To run the training and evaluation just run the `python run.py`. You will see progress of the training on the training set.
Afterwards it will be evaluated on the test set and a single real number will be printed. It stands for the error on the test set.

# Training your own network
If you wish to train your own network, have a look at the `model.py` file and the `create_net` method. You can redefine the neural network there. Also have a look at the `compile_model` method, where you specify algorithms that will minimize loss function of your training algorithm.

# Tweaking the training
You can play around with some parameters in the `run.py` script. 
`model.fit(train_x, train_y, epochs=5, batch_size=32, verbose=1)`
`epochs` stands for number of times the learning algorithm processes all the training data
`batch_size` stands for the number of examples process at once during the training
`verbose` shows overall progress if set to `1`.

`dataset = Dataset(shuffle=True, normalize=True, subtract_mean=True)`
It is good practice to shuffle data, scale them to [0,1] interval and subtract mean image from both training and testing examples. However, you can turn it off at will and see if it affects the results.

# Method used
I used a simple convolutional network  [see origins here](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) with [relu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) and [softmax](https://en.wikipedia.org/wiki/Softmax_function) activation functions. Relu has proven very effective in deep learning and softmax serves as a non-max supression method, which is useful when representing classification examples as binary vectors. The training is performed by [ADAM](https://arxiv.org/abs/1412.6980) algorithm.
