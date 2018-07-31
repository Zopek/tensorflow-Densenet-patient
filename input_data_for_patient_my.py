# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import time

print('for 1')

def get_train_dir(filepath):

	dirs = os.listdir(filepath)
	dirs = np.array(dirs[:-3000])
	perm0 = np.arange(len(dirs))
	np.random.shuffle(perm0)
	dirs = dirs[perm0]

	return dirs

def get_test_dir(filepath):

	dirs = os.listdir(filepath)
	dirs = np.array(dirs[-3000:])
	perm0 = np.arange(len(dirs))
	np.random.shuffle(perm0)
	dirs = dirs[perm0]

	return dirs

def get_size(filepath):

	test_size = len(get_test_dir(filepath))
	train_size = len(get_train_dir(filepath))

	return train_size, test_size

# 多标签转换
def label_convert_mul(label):

	labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
	labels[int(label[0])] = 1
	labels[int(label[1]) + 4] = 1
	labels[int(label[2]) + 7] = 1
	labels[int(label[3]) + 11] = 1

	return labels.reshape((1, -1)) 

# 单一标签转换
def label_convert(label):

	labels = np.array([0, 0, 0, 0])
	labels[int(label[0])] = 1

	return labels.reshape((1, -1)) 

def train_next_batch(filepath, step, batch_size):

	dirs = get_train_dir(filepath)
	current_num = step * batch_size

	for filename in dirs[current_num:(current_num + batch_size)]:
		image = np.load(filepath + '/' + filename + '/image.npy')
		image = image.reshape((1, -1))
		if filename == dirs[current_num]:
			images = image
		else:
			images = np.vstack((images, image))

		label = np.load(filepath + '/' + filename + '/label.npy')
		label = label_convert(label)
		if filename == dirs[current_num]:
			labels = label
		else:
			labels = np.vstack((labels, label))

	return images, labels

def test_next_batch(filepath, step):

	dirs = get_test_dir(filepath)
	current_num = step * 300
	for filename in dirs[current_num:(current_num + 300)]:
		image = np.load(filepath + '/' + filename + '/image.npy')
		image = image.reshape((1, -1))
		if filename == dirs[current_num]:
			images = image
		else:
			images = np.vstack((images, image))

		label = np.load(filepath + '/' + filename + '/label.npy')
		label = label_convert(label)
		if filename == dirs[current_num]:
			labels = label
		else:
			labels = np.vstack((labels, label))

	return images, labels