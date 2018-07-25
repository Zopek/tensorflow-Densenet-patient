# -*- coding: UTF-8 -*-

import numpy as np
import os

filepath = '/DATA/data/qyzheng/patient_image_4/train/synthetic'
label = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

for i in range(3):

	dirs = os.listdir(filepath + '/' + str(i))
	for filename in dirs:
		
		np.save((filepath + '/' + str(i) + '/' + filename + '/label.npy'), label[i])