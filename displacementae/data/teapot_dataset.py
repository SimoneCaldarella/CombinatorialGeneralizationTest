#!/usr/bin/env python3
# Copyright 2021 Hamza Keurti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title          :displacementae/data/teapot_dataset.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :10/09/2022
# @version        :1.0
# @python_version :3.7.4
"""
Dataset of a 3D object in different orientations.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The module :mod:`data.teapot` contains a data handler for a dataset 
generated from the teapot model
`dsprites dataset <https://github.com/deepmind/dsprites-dataset>`.
"""

import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
import os
import h5py

import data.transition_dataset as trns_dataset




class TeapotDataset(trns_dataset.TransitionDataset):
    def __init__(self, root, rseed=None, transitions_on=True,
                 n_transitions: int = None,
                 num_train=200,
                 num_val: int = 30):
        super().__init__(rseed, transitions_on, n_transitions)

        # Read Data from file.
        self._root = root
        self._images, self._transitions = self._process_hdf5()

        # Number of samples
        self.num_train = num_train
        self.num_val = num_val
        
        n = self._images.shape[0]
        if  n < (num_train+num_val):
            print(f"Not enough samples {n} for chosen " + 
                  f"--num_train {num_train} and --num_val {num_val}")
            self.num_val = 20
            self.num_train = self._images.shape[0] - self.num_val
            print(f"Reset num_train to {self.num_train} " + 
                  f"and num_val to {self.num_val}")
            



        data = {}
        data["in_shape"] = self._images.shape[2:]
        data["action_shape"] = [self._transitions.shape[-1]]
        self.action_dim = self._transitions.shape[-1]
        self._data = data



    def _process_hdf5(self):
        """
        opens the hdf5 dataset file.
        """
        filepath = os.path.join(self._root)
        self._file = h5py.File(filepath,'r')
        images = self._file['images'][()]
        transitions = self._file['rotations'][()] 
        # images = self._file['images']
        # transitions = self._file['rotations']
        return images, transitions


    def __len__(self):
        return self.num_train


    def __getitem__(self, idx):
        images = self._images[idx]
        dj = self._transitions[idx,1:]
        return images, [], dj


    @property
    def n_actions(self):
        """
        Number of all possible discrete actions.
        """


    def get_example_actions(self):
        a = np.zeros((self.action_dim*2+1,self.action_dim))
        for i in range(self.action_dim):
            a[1+2*i:3+2*i,i] = np.array([1,-1])
        
        # if self.rotate_actions:
        #     rot_a = a.copy()
        #     rot_a[...,:2] = rot_a[...,:2] @ self._rot_mat
        #     return rot_a, a
        # else:
        return a, a

    @property
    def in_shape(self):
        return self._data["in_shape"]

    @property
    def action_shape(self):
        return self._data["action_shape"]

    def get_val_batch(self):
    #     imgs = self._images[self.num_train:self.num_train+self.num_val]
    #     transitions = self._transitions[self.num_train:self.num_train+self.num_val]
        return self[self.num_train:self.num_train+self.num_val]

if __name__ == '__main__':
    pass