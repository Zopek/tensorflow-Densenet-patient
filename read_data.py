# -*- coding: UTF-8 -*-

import tensorflow as tf 

def read_and_decode(filename):


	filename_queue = tf.train.string_input_producer([filename])

	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example, features={
		'label': tf.FixedLenFeature([], tf.string),
		'image': tf.FixedLenFeature([], tf.string)
		})

	image = tf.decode_raw(features['image'], tf.float32)
	image = tf.reshape(image, [224, 224, 1])
	label = tf.cast(features['label'], tf.float32)

	return image, label

def next_batch(filename, batch_size):

	image, label = read_and_decode(filename)
	image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=batch_size*5, min_after_dequeue=batch_size*3)

	return  image_batch, label_batch






