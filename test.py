# -*- coding: UTF-8 -*-

# import tensorflow as tf

import input_data_for_patient as input_data
import numpy as np

filepath = '/DATA/data/hyguan/liuyuan_spine/data_all/patient_image_4'
train_size, test_size = input_data.get_size(filepath)
train_dirs = input_data.get_train_dir(filepath)
test_dirs = input_data.get_test_dir(filepath)

print((train_size + test_size))
labels = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

for filename in train_dirs:

	label = np.load(filepath + '/' + filename + '/label.npy')
	for i in range(4):
		labels[i][int(label[i])] += 1

	if filename == train_dirs[10000]:
		print('10000')
	if filename == train_dirs[30000]:
		print('30000')
	if filename == train_dirs[50000]:
		print('50000')
	if filename == train_dirs[70000]:
		print('70000')
	if filename == train_dirs[90000]:
		print('90000')

for filename in test_dirs:

	label = np.load(filepath + '/' + filename + '/label.npy')
	for i in range(4):
		labels[i][int(label[i])] += 1
	if filename == test_dirs[10000]:
		print('10000')

print(labels)
"""
for filename in train_dirs:

	label = np.load(filepath + '/' + filename + '/label.npy')
	labels = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
	for i in range(4):
		labels[i][int(label[i])] += 1

for filename in test_dirs:

	label = np.load(filepath + '/' + filename + '/label.npy')
	labels = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
	for i in range(4):
		labels[i][int(label[i])] += 1

print(labels)
"""
"""
# -*- coding: UTF-8 -*-

import numpy as np
import os

def get_dir(filepath):

	dirs = os.listdir(filepath)

	return dirs

filepath = '/DATA/data/qyzheng/patient_image_4/1'
dirs = get_dir(filepath)

labels = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
for filename in dirs:

	label = np.load(filepath + '/' + filename + '/label.npy')
	for i in range(4):
		labels[i][int(label[i])] += 1

print(labels)
"""