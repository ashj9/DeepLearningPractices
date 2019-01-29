import numpy as np
import matplotlib  as mp
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.examples.tutorials.mnist import input_data
import math

def _variable_on_cpu(name, shape, initializer, use_float16=False):
	with tf.device("/cpu:0"):
		dtype = tf.float16 if use_float16 else tf.float32
		var = tf.get_variable(name, shape, initializer = initializer, dtype = dtype)
	return var


def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier = True):
	if use_xavier:
		initializer = tf.contrib.layers.xavier_initializer()
	else:
		initializer = tf.truncated_normal_initializer(stddev = stddev)

	var = _variable_on_cpu(name, shape, initializer)
	# wd : weight decay is a float value
	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		# tf.add_to_collection('losses', weight_decay)
	return var

def conv2d(
	inputs,
	num_output_channels,
	kernel_size,
	scope,
	stride = [1,1],
	padding = 'SAME',
	use_xavier = True,
	stddev = 1e-3,
	weight_decay = 0.0,
	activation_fn = tf.nn.relu,
	bn = False,
	bn_decay = None,
	is_training = None):
	with tf.variable_scope(scope) as sc:
		kernel_h , kernel_w = kernel_size
		num_in_channels = inputs.get_shape()[-1].value
		kernel_shape = [kernel_h, kernel_w, num_in_channels, num_output_channels]
		kernel = _variable_with_weight_decay('weights',
			shape = kernel_shape,
			stddev = stddev,
			wd = weight_decay,
			use_xavier = use_xavier)

		stride_h, stride_w = stride
		stride_shape = [1, stride_h, stride_w, 1]

		outputs = tf.nn.conv2d(inputs,
			kernel,
			stride_shape,
			padding = padding)
		biases = _variable_on_cpu('biases', [num_output_channels],tf.constant_initializer(0.0))

		outputs = tf.nn.bias_add(outputs, biases)
		if bn:
			outputs = batch_norm_for_conv2d(outputs, is_training, bn_decay =bn_decay, scope = 'bn')
		if activation_fn is not None:
			outputs = activation_fn(outputs)

		return outputs


def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope):
	'''
	Args:
	inputs: Tensor 4D, BHWC input maps
	is_training : boolean tf.Variable , true indicates that training phase
	bn_decay : float or float tensor variable, controling the moving average weight
	scope: string, varible scope
	Return:
	normed: batch-normalized maps
	'''

	return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay)

def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
	'''
	Args:
	inputs: Tensor 4D, BHWC input maps
	is_training : boolean tf.Variable , true indicates that training phase
	bn_decay : float or float tensor variable, controling the moving average weight
	scope: string, varible scope
	moments_dims: a list if ints, indicating dimensions for moments calculation
	Return:
	normed: batch normalized maps
	'''
	with tf.variable_scope(scope) as sc:
		num_channels = inputs.get_shape()[-1].value
		beta = tf.Variable(tf.constant(0.0, shape=[num_channels]), 
			name = 'beta',
			trainable = True
			)
		gamma = tf.Variable(tf.contant(1.0, shape=[num_channels]),
			name='gamma',
			trainable = True
			)
		batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
		decay = bn_decay if bn_decay is not None else 0.9
		ema = tf.train.ExponentialMovingAverage(decay=decay)

		#operator that maintains the moving averages of variables
		ema_apply_op = tf.cond(is_training,
			lambda:ema.apply([batch_mean, batch_var]),
			lambda:tf.no_op())

		def mean_var_with_update():
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		#ema.average returns the Variable holding the average of var
		mean, var = tf.cond(is_training,
			mean_var_with_update,
			lambda: (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
	return normed

def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D max pooling.
  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.max_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs



mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, 784], name='x_in')
true_y = tf.placeholder(tf.float32, [None, 10], name='y_in')

keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x, ([-1,28,28,1]))
print(np.array(x_image).shape)


hidden_1 = conv2d(x_image, 
	num_output_channels = 5,
	kernel_size = [3,3],
	scope = 'conv_1st',
	stride = [1,1],
	padding = 'VALID'	
	)

# pool_1 = max_pool2d(inputs = hidden_1,
# 	kernel_size = [2,2],
# 	scope ='maxpool_1',
# 	stride = [2,2],
# 	padding = 'VALID'
# 	)


hidden_2 = conv2d(hidden_1, 
	num_output_channels = 11,
	kernel_size = [5,5],
	stride = [1,1],
	padding = 'VALID',
	scope = 'conv_2nd'
	)

pool_2 = max_pool2d(inputs = hidden_2,
	kernel_size = [2,2],
	scope='maxpool_2',
	stride = [2,2],
	padding = 'VALID',
	)

hidden_3 = conv2d(pool_2, 
	num_output_channels = 15,
	kernel_size = [7,7],
	stride = [1,1],
	padding = 'VALID',
	scope = 'conv_3rd'
	)

hidden_3 = tf.nn.dropout(x = hidden_3,
	keep_prob = 0.85
	)

out_y = tf.contrib.layers.fully_connected(tf.layers.flatten(hidden_3), 10, activation_fn = tf.nn.softmax)
cross_entropy = -tf.reduce_sum(true_y * tf.log(out_y))

correct_prediction = tf.equal(tf.argmax(out_y, 1), tf.argmax(true_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

batch_size = 50
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# by this time, all global variables might have been initialized

for i in range(2000):
	batch = mnist.train.next_batch(batch_size)
	sess.run(train_step, feed_dict={x:batch[0], true_y:batch[1], keep_prob: 0.5})
	if i % 100 == 0 and i!= 0:
		trainAccuracy = sess.run(accuracy, feed_dict={x:batch[0], true_y:batch[1], keep_prob : 1.0})
		print("Step %d, training accuracy is %g"%(i, trainAccuracy))

test_accuracy = sess.run(accuracy, feed_dict = {x:mnist.test.images,
	true_y:mnist.test.labels, keep_prob:1.0})
print("Test Accuracy: %g"%(test_accuracy))


def get_activations(layer, stimuli):
	units = sess.run(layer, feed_dict={x:np.reshape(stimuli, [1,784], order='F'),
		keep_prob : 1.0})
	plotNNFilters(units)

def plotNNFilters(units):
	filters = units.shape[3]
	plt.figure(1, figsize = (20,20))
	n_columns = 6
	n_rows = math.ceil(filters / n_columns) + 1
	for i in range(filters):
		plt.subplot(n_rows, n_columns, i+1)
		plt.title('Filter ' + str(i))
		plt.imshow(units[0,:,:,i], interpolation = "nearest", cmap = 'gray')
	plt.show()
imageToUse = mnist.test.images[0]
plt.imshow(np.reshape(imageToUse, [28,28]), interpolation="nearest", cmap="gray")

plt.show()
get_activations(hidden_1, imageToUse)
get_activations(hidden_2, imageToUse)
get_activations(hidden_3, imageToUse)
