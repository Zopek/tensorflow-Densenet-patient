# -*- coding: UTF-8 -*-

import os
import tensorflow as tf 
import numpy as np

filepath = '/DB/rhome/qyzheng/Desktop/qyzheng/patient_image_4/train'


def get_dir(filepath):

	for h in range(4):
		dirs = os.listdir(filepath + '/' + str(h))
		for i in range(len(dirs)):
			if h == 0 and i == 0:
				dirs_list = [str(h) + '/' + dirs[i]]
			else:
				dirs_list.append(str(h) + '/' + dirs[i])

	return dirs_list

def label_convert(label):

	if int(label[0]) == 0:
		labels = np.array([1, 0])
	else:
		labels = np.array([0, 1])

	return labels.reshape((1, -1)) 

writer = tf.python_io.TFRecordWriter(filepath + '/train.tfrecords')
dirs = get_dir(filepath)
for filename in dirs:

	image = np.load(filepath + '/' + filename + '/image.npy')
	image = image.reshape((1, -1))
	image_bytes = image.tobytes()

	label = np.load(filepath + '/' + filename + '/label.npy')
	label = label_convert(label)
	label_bytes = label.tobytes()

	example = tf.train.Example(features=tf.train.Features(feature={
		"label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_bytes])),
		"image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
		}))

	writer.write(example.SerializeToString())
writer.close()