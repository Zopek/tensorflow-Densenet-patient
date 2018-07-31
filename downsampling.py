# -*- coding: UTF-8 -*-

import numpy as np
import os
import shutil
import random

def get_dir(filepath):

	for h in range(1,4):
		dirs = os.listdir(filepath + '/' + str(h))
		for i in range(len(dirs)):
			dirs_next = os.listdir(filepath + '/' + str(h) + '/' + dirs[i])
			for j in range(len(dirs_next)):
				if h == 1 and i == 0 and j == 0:
					dirs_list = [str(h) + '/' + dirs[i] + '/' + dirs_next[j]]
					name_list = [dirs_next[j]]
				else:
					dirs_list.append(str(h) + '/' + dirs[i] + '/' + dirs_next[j])
					name_list.append(dirs_next[j])

	return dirs_list, name_list

filepath = '/DATA/data/hyguan/liuyuan_spine/data_all/patient_image_4'
dirs, name = get_dir(filepath)
print(len(dirs))

for i in range(len(dirs)):

	label = np.load(filepath + '/' + dirs[i] + '/label.npy')

	if int(label[0]) == 0:
		if random.random() <= 0.107568:
			oldpath = filepath + '/' + dirs[i]
			newpath = '/DB/rhome/qyzheng/Desktop/qyzheng/patient_image_4/train/0/' + name[i]
			shutil.copytree(oldpath,newpath)
	"""

	if int(label[0]) == 3:
		oldpath = filepath + '/' + dirs[i]
		newpath = '/DATA/data/qyzheng/patient_image_4/train/3/' + name[i]
		shutil.copytree(oldpath,newpath)
	"""
	if i%5000 == 0:
		print(i, ' ', label)
