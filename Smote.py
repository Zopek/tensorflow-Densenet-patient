# -*- coding: UTF-8 -*-

import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
import input_data_for_patient as input_data

class Smote:
    def __init__(self,samples,N=100,k=5):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0
       # self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))

    def over_sampling(self):
        N=int(self.N/100)
        # self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        print('neighbors',neighbors)
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            #print nnarray
            self._populate(N,i,nnarray)
        # return self.synthetic


    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn=random.randint(0,self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            synthetic=self.samples[i]+gap*dif
            filename = '/DATA/data/qyzheng/patient_image_4/synthetic/0/synthetic_0_' + str(self.newindex)
            os.makedirs(filename)
            numpy.save('image.npy', synthetic)
            self.newindex+=1

filepath = '/DATA/data/hyguan/liuyuan_spine/data_all/patient_image_4'
filepath_my = '/DATA/data/qyzheng/patient_image_4/synthetic/0'
dirs = get_train_dir(filepath)
size = 0

for i in range(len(dirs)):

    label = np.load(filepath + '/' + dirs[i] + '/label.npy')
    if int(label[0]) == 1:
        image = np.load(filepath + '/' + dirs[i] + '/image.npy')
        image = image.reshape((1, -1))
        if size == 0:
            images = image
            size += 1
        else:
            images = np.vstack((images, image))
            size += 1

print('size', size)
s=Smote(images,N=200)