from dataset import Dataset
from model import create_net, compile_model

#obtain and prepare data
dataset = Dataset(shuffle=True, normalize=True, subtract_mean=True)

#define neural network architecture
model = create_net(dataset.example_input_shape(), dataset.num_classes())

#choose training algorithm
compile_model(model)

#train model
train_x, train_y = dataset.get_training_data()
model.fit(train_x, train_y, epochs=5, batch_size=32, verbose=1)

#evaluate model
test_x, test_y = dataset.get_testing_data()
print(1-model.evaluate(test_x, test_y, verbose=0)[1])

