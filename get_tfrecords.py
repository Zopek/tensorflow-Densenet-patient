# -*- coding: UTF-8 -*-

import os
import tensorflow as tf 
import numpy as np

filepath = '/DB/rhome/qyzheng/Desktop/qyzheng/patient_image_4/test'


def get_dir(filepath):

	for h in range(0, 4):
		dirs = os.listdir(filepath + '/' + str(h))
		for i in range(len(dirs)):
			if h == 0 and i == 0:
				dirs_list = [str(h) + '/' + dirs[i]]
			else:
				dirs_list.append(str(h) + '/' + dirs[i])

	dirs_list = np.array(dirs_list)
	perm0 = np.arange(len(dirs_list))
	np.random.shuffle(perm0)
	dirs_list = dirs_list[perm0]

	return dirs_list

def label_convert(label):

	if int(label[0]) == 0:
		labels = np.array([1, 0])
	else:
		labels = np.array([0, 1])
	labels = labels.astype(np.float32)

	return labels.reshape((1, -1)) 

def multi_label_convert(label):

	labels = np.array([0, 0, 0, 0])
	labels[int(label[0])] = 1
	labels = labels.astype(np.float32)

	return labels.reshape((1, -1))

writer = tf.python_io.TFRecordWriter(filepath + '/test.tfrecords')
dirs = get_dir(filepath)

i = 0
for filename in dirs:

	image = np.load(filepath + '/' + filename + '/image.npy')
	image = image.reshape((1, -1))
	image_bytes = image.tobytes()

	label = np.load(filepath + '/' + filename + '/label.npy')
	label_mul = multi_label_convert(label)
	label_mul_bytes = label_mul.tobytes()
	label = label_convert(label)
	label_bytes = label.tobytes()

	i += 1
	if i % 100 == 0:
		print(image.shape, ' ', image)
		print(label_mul.shape, ' ', label_mul)
		print(label.shape, ' ', label)

	example = tf.train.Example(features=tf.train.Features(feature={
		"label_mul": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_mul_bytes])),
		"label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_bytes])),
		"image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
		}))

	writer.write(example.SerializeToString())
writer.close()