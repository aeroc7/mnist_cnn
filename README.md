# mnist_cnn
Implements a Convolutional Neural Network to classify handwritten digits from the MNIST dataset.

# Network structure
There is 5 total neural layers in this network:
- A conv input layer, taking in 1 channel (MNIST data is grayscale, hence only 1 channel)
- Hidden conv layer 1, which takes in 16 channels
- Hidden conv layer 2, which takes in 64 channels
- Hidden conv layer 3, which takes in 85 channels
- A fully connected layer for classification (10 outputs, digits 0 - 9 in the MNIST set)

All of the convolutional layers use the ReLU activation function, and all of the convolutional layers implement dropout as well.

The first two neural layers (input layer and hidden layer 1) have max pooling layers.

# Results
The hyperparameters used for the network training were:
- 65 epochs
- Initial learning rate of 0.001
- Batch size of 64
- Dropout chance of 25%

I also used a learning rate scheduler to reduce the learning rate after several epochs with no decrease in loss.
- Factor of 0.1
- Patience of 3 epochs

With all this, I was able to achieve 100% accuracy on the training set, and 99.51% on the test set.

# Improvements
I am sure that with the implementation of data augmentation, the accuracy of the network will continue to improve.
