from keras.datasets import mnist

(train_images, train_labels),(test_images,test_labels) = mnist.load_data()
train_images.shape

from keras import models
from keras import layers

# Create the network
network = models.Sequential()
# Adding the first layer fully connected
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
# Adding the second layer fully connected
network.add(layers.Dense(10, activation='softmax'))
# Before running we need to specify also the optimizer (way for autoimprove the network), loss function (way to measure performances), matrics to monitor
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

# Converting the 0-255 to interval 0-1 easier to be processed
train_images = train_images.reshape((60000, 28*28))
train_images[0]
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28*28))
test_images[0]
test_images = test_images.astype('float32')/255

# Categorically encode the labels
from keras.utils import to_categorical

# 0 or 1 it depends from the value T/F
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Train the network is done via the fit method
#We have two metrics: the loss of the network and the accuracy over the training
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# used to evaluate the model on test data
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:',test_acc)
