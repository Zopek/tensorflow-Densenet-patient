# -*- coding: UTF-8 -*-

import tensorflow as tf 

def read_and_decode(filename, num_epoch):

	filename_queue = tf.train.string_input_producer([filename], shuffle=False, num_epochs=num_epoch)

	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example, features={
		'label_mul': tf.FixedLenFeature([], tf.string),
		'label': tf.FixedLenFeature([], tf.string),
		'image': tf.FixedLenFeature([], tf.string)
		})

	image = tf.decode_raw(features['image'], tf.float32)
	image = tf.reshape(image, [50176,])
	#print(image.shap
	label_mul = tf.decode_raw(features['label_mul'], tf.float32)
	print('label_mul is', type(label_mul))
	label_mul = tf.reshape(label_mul, [4,])
	label = tf.decode_raw(features['label'], tf.float32)
	print('label is', type(label))
	label = tf.reshape(label, [2,])

	#print(type(image))

	return image, label, label_mul

def next_batch(filename, batch_size, num_epoch):

	image, label, label_mul = read_and_decode(filename, num_epoch)
	print("decode OK")
	num_threads = 32
	min_after_dequeue = 10
	capacity = num_threads * batch_size + min_after_dequeue

	image_batch, label_batch, label_mul_batch = tf.train.shuffle_batch(
		[image, label, label_mul], 
		batch_size=batch_size, 
		num_threads=num_threads,
		capacity=capacity, 
		min_after_dequeue=min_after_dequeue)
	print("shuffle OK")

	return  image_batch, label_batch, label_mul_batch






