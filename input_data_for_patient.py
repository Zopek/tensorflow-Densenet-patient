# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import time

def get_train_dir(filepath):

	for h in range(1,4):
		dirs = os.listdir(filepath + '/' + str(h))
		for i in range(len(dirs)):
			dirs_next = os.listdir(filepath + '/' + str(h) + '/' + dirs[i])
			for j in range(len(dirs_next)):
				if h == 1 and i == 0 and j == 0:
					dirs_list = [str(h) + '/' + dirs[i] + '/' + dirs_next[j]]
				else:
					dirs_list.append(str(h) + '/' + dirs[i] + '/' + dirs_next[j])

	return dirs_list

def get_test_dir(filepath):

	dirs = os.listdir(filepath + '/4')
	for i in range(len(dirs)):
		dirs_next = os.listdir(filepath + '/4' + '/' + dirs[i])
		for j in range(len(dirs_next)):
			if i == 0 and j == 0:
				dirs_list = ['4/' + dirs[i] + '/' + dirs_next[j]]
			else:
				dirs_list.append('4/' + dirs[i] + '/' + dirs_next[j])

	return dirs_list

def get_size(filepath):

	train_dirs = get_train_dir(filepath)
	test_dirs = get_test_dir(filepath)

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

def train_next_batch(filepath, step, batch_size):

	dirs = get_train_dir(filepath)
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