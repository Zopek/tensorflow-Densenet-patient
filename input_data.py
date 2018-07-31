# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import time

print('for synthetic')

def get_dir(filepath):

	for h in range(4):
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

def get_train_dir(dirs_list):

	dirs_list = dirs_list[:-8000]
	perm0 = np.arange(len(dirs_list))
	np.random.shuffle(perm0)
	dirs_list = dirs_list[perm0]

	return dirs_list

def get_test_dir(dirs_list):

	dirs_list = dirs_list[-8000:]
	perm0 = np.arange(len(dirs_list))
	np.random.shuffle(perm0)
	dirs_list = dirs_list[perm0]

	return dirs_list

def get_size(dirs_list):

	train_dirs = get_train_dir(dirs_list)
	test_dirs = get_test_dir(dirs_list)

	return len(train_dirs), len(test_dirs)

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

def train_next_batch(filepath, step, batch_size, dirs_list):

	dirs = get_train_dir(dirs_list)
	size = len(dirs)
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

def test_next_batch(filepath, step, dirs_list):

	dirs = get_test_dir(dirs_list)
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