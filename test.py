# -*- coding: UTF-8 -*-

# import tensorflow as tf
import os
import numpy as np

def get_dir(filepath):

	for h in range(4):
		dirs = os.listdir(filepath + '/' + str(h))
		for i in range(len(dirs)):
			if h == 0 and i == 0:
				dirs_list = [str(h) + '/' + dirs[i]]
			else:
				dirs_list.append(str(h) + '/' + dirs[i])

	dirs_list = np.array(dirs_list)
	print(dirs_list[-10:])

	perm0 = np.arange(len(dirs_list))
	np.random.shuffle(perm0)
	dirs_list = dirs_list[perm0]
	print(dirs_list[-10:])

	return dirs_list

filepath = '/DB/rhome/qyzheng/Desktop/qyzheng/patient_image_4/train/synthetic'

dirs = get_dir(filepath)

"""
import input_data_for_patient as input_data
import numpy as np

filepath = '/DATA/data/hyguan/liuyuan_spine/data_all/patient_image_4'
train_size, test_size = input_data.get_size(filepath)
dirs = input_data.get_train_dir(filepath)

labels = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

for filename in dirs:

	label = np.load(filepath + '/' + filename + '/label.npy')
	for i in range(4):
		labels[i][int(label[i])] += 1


print(labels)
"""
"""
import input_data_for_patient_my as input_data
import numpy as np

filepath = '/DATA/data/qyzheng/patient_image_4/1'
train_size, test_size = input_data.get_size(filepath)
dirs = input_data.get_dir(filepath)

print((train_size + test_size))
labels = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

for filename in dirs:

	label = np.load(filepath + '/' + filename + '/label.npy')
	for i in range(4):
		labels[i][int(label[i])] += 1


print(labels)
"""
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