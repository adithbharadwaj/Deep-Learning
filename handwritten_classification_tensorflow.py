import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data # for loading the mnist dataset

import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("mnist_data/", one_hot= True) 
# one_hot stands for the nature of output. i,e we have 10 classes and if one_hot is true, 
# output is of the form [0,0,1,0,0,0,0,0,0,0], where 1 indicated the ans, which, in this case is 2 (0-9) 		


n_hidden_1 = 500  # number of neurons in the hidden layer 1
n_hidden_2 = 500  # nuber of neurons in hidden layer 2
n_hidden_3 = 500  # neurons in hidden layer 3.

n_classes = 10  # number of outputs. i.e number of neurons in the output layer

# height x width (here height is None)
x = tf.placeholder('float', [None, 784]) # placeholder for the input which is in the form of a 28x28 matrix that has been 
										 # flattened out into a single dimentional 784 sized tensor. 
y = tf.placeholder('float', [None, 10]) # for labels.


def forward_propagation(data):

	# random initialization of weights and biases for different layers.
	# we are creating a tensorflow variable in the form of tensors of a particular shape (depends on the number of neurons)

	hidden_1 = {'weights': tf.Variable(tf.random_normal([784, n_hidden_1])), 
			    'biases': tf.Variable(tf.random_normal([n_hidden_1]))}

	hidden_2 = {'weights': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
			    'biases': tf.Variable(tf.random_normal([n_hidden_2]))}

	hidden_3 = {'weights': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
			    'biases': tf.Variable(tf.random_normal([n_hidden_3]))}

	output_layer = {'weights': tf.Variable(tf.random_normal([n_hidden_3, n_classes])),
			    	  'biases': tf.Variable(tf.random_normal([n_classes]))}

	# forward propagation

	# (input * weights) + biases.
	# the add function is used to add tensors and matmul stands for matrux multiplication.

	l1 = tf.add(tf.matmul(data, hidden_1['weights']), hidden_1['biases'])
	l1 = tf.nn.relu(l1)  # using the relu activation function (rectified linear unit) tf.nn.sigmoid() can alse be used.

	l2 = tf.add(tf.matmul(l1, hidden_2['weights']), hidden_2['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3['weights']), hidden_3['biases'])
	l3 = tf.nn.relu(l3)
	
	output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

	return output


def train(x):

	batch_size = 100

	prediction = forward_propagation(x)

	# here, reduce mean is similar to np.mean(), it is similar to multiplying it by 1/(2*m).

	# softmax is an activation function similar to sigmoid or relu
	# cross_entropy is basically the logistic regression cost function y*log(h(x)) + (1 - y)*log(1 - h(x))

	# here, we are computing the cost using cross entropy.
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction, labels = y))

	# this is the gradient descent optimization algorithm that performs gradient descent and minimizes the cost.
	# there are many other optimization algos in the train module.
	# learning_rate = 0.001 by default. it can be changed: GradientDescentOptimizer(learning_rate = 0.1)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(cost)

	epochs = 10 # number of iterations.

	with tf.Session() as sess:  # to open and close sessions.

		sess.run(tf.initialize_all_variables()) # we need to initialize the variables.

		for i in range(epochs):
			loss = 0
			for j in range(int(mnist.train.num_examples/batch_size)): # we are dividing the 60,000 examples into batches of size 100.

				batch_x, batch_y = mnist.train.next_batch(batch_size) # splitting the data into 
																	  # batches using tensorflow's built in function.

				gradient, value = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y}) 
				# we are feeding the values of x, y for cost 
				# which takes 2 arguments.

				loss += value

			print('loss after epoch ', i, loss)

		#calculating the accuracy.	
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('accuracy is', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))	

		# predicting the values of the test dataset. 

		p = tf.argmax(prediction, 1)
		pred = p.eval({x:mnist.test.images})

		return pred

if __name__ == '__main__':

	pred = train(x)

	# plotting the test dataset sample as a greyscale image
	pixels = np.array(mnist.test.images[0]) # choosing an image to display
	pixels = pixels.reshape((28, 28)) # reshaping it because it is flattened into a 1d tensor of 784 values.  
	plt.imshow(pixels, cmap='gray') 
	plt.show()

	print("predicted : ", pred[0]) # printing the value of the corresponding prediction
