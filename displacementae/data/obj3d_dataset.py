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
# @title          :displacementae/data/obj3d_dataset.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :10/09/2022
# @version        :1.0
# @python_version :3.7.4
"""
Dataset of a 3D object in different orientations.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The module :mod:`data.obj3d` contains a data handler for a hdf5 dataset 
generated from .obj models.
"""

import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
import os
import h5py

import data.transition_dataset as trns_dataset




class Obj3dDataset(trns_dataset.TransitionDataset):
    def __init__(self, root, rseed=None, transitions_on=True,
                 n_transitions: int = None,
                 num_train=200,
                 num_val: int = 30,
                 resample:bool=False,
                 num_samples:int=200,
                 normalize_actions:bool=False):
        super().__init__(rseed, transitions_on, n_transitions)

        # Read Data from file.
        self._root = root
        self.resample = resample
        self.num_samples = num_samples
        self.normalize_actions = normalize_actions

        # Number of samples
        self.num_train = num_train
        self.num_val = num_val
        
        self.load_data()
        self.load_attributes()
        self.sample_val_batch()
        
        self.rots_idx = np.array([])
        self.trans_idx = np.array([])
        self.col_idx = np.array([])
        
        if (not self.translate_only):
            self.rots_idx = np.arange(3)
        
        if self.translate:
            self.trans_idx = np.arange(start=3,stop=6)
        if self.translate_only:
            self.trans_idx = np.arange(3)
        if self.color:
            self.col_idx = np.array(self._transitions.shape[-1]-1)        
        
        rng = self.rots_range[1] - self.rots_range[0]
        if self.mode=='continuous':
            self.rots_stepsize=rng/4
        else:
            self.rots_stepsize=rng/(self.rots_n_values-1)
        

        data = {}
        data["in_shape"] = self._images.shape[2:]
        data["action_shape"] = [self._transitions.shape[-1]]
        self.action_dim = self._transitions.shape[-1]
        self._data = data




    def load_data(self):
        """
        Loads samples from an hdf5 dataset.
        """
        filepath = os.path.join(self._root)
        if self.resample:
            self.resample_data()
        else:
            with h5py.File(filepath,'r') as f:
                self._images = f['images'][:self.num_train]
                self._transitions = f['actions'][:self.num_train,1:] 
                if self.normalize_actions:
                    self.M = np.abs(self._transitions).max(axis=(0,1))
                    self._transitions /= self.M
            # images = self._file['images']
            # transitions = self._file['rotations']

    def resample_data(self):
        """
        Replaces new samples in memory.
        """
        if self.resample:
            if hasattr(self,"_images"):
                del self._images
                del self._transitions
            indices = np.sort(
                    np.random.choice(
                        self.num_train,size=self.num_samples, replace=False))
            filepath = os.path.join(self._root)
            with h5py.File(filepath,'r') as f:
                self._images = f['images'][indices]
                self._transitions = f['actions'][indices,1:]
                if self.normalize_actions:
                    self.M = np.abs(self._transitions).max(axis=(0,1))
                    self._transitions /= self.M
        else:
            pass 
    
    def load_attributes(self):
        """
        Loads the atributes of the dataset
        """
        with h5py.File(self._root,'r') as f:
            self.attributes_dict = dict(f['images'].attrs)
        #         "obj_filename":obj_filename,  
        # "figsize":figsize,
        # "dpi":dpi, 
        # "lim":lim,
        self.mode = self.attributes_dict["mode"] 
        self.translate=self.attributes_dict["translate"]
        self.translate_only=self.attributes_dict["translate_only"]
        self.rots_range=self.attributes_dict["rots_range"]
        self.n_steps=self.attributes_dict["n_steps"] 
        self.n_samples=self.attributes_dict["n_samples"]
        self.color= self.attributes_dict["color"]
        self.rots_n_values=self.attributes_dict["n_values"] 
        if self.translate or self.translate_only:
            self.trans_grid=self.attributes_dict["translation_grid"]
            self.trans_stepsize=self.attributes_dict["translation_stepsize"]
            self.trans_range=self.attributes_dict["translation_range"]
        

    def sample_val_batch(self):
        filepath = os.path.join(self._root)
        nt = self.num_train
        nv = self.num_val
        with h5py.File(filepath,'r') as f:
            n = f['images'].shape[0]
            if  n < (nt+nv):
                raise ValueError(f"Not enough samples {n} for chosen " + 
                    f"--num_train={nt} and --num_val={nv}")
            self.val_imgs = f['images'][nt:nt+nv]
            self.val_actions = f['actions'][nt:nt+nv,1:]
            if self.normalize_actions:
                self.val_actions /= self.M


    def __len__(self):
        if self.resample:
            return self.num_samples
        else:
            return self.num_train


    def __getitem__(self, idx):
        images = self._images[idx]
        dj = self._transitions[idx]
        return images, [], dj


    @property
    def n_actions(self):
        """
        Number of all possible discrete actions.
        """


    def get_example_actions(self):
        a = np.zeros((self.action_dim*2+1,self.action_dim))
        for i in range(self.action_dim):
            if i in self.rots_idx:
                a[1+2*i:3+2*i,i] = np.array([1,-1])*self.rots_stepsize
            elif i in self.trans_idx:
                a[1+2*i:3+2*i,i] = np.array([1,-1])*self.trans_stepsize
            else:
                a[1+2*i:3+2*i,i] = np.array([1,-1])
            a_in = a.copy()
            if self.normalize_actions:
                a_in /= self.M
        return a_in, a

    @property
    def in_shape(self):
        return self._data["in_shape"]

    @property
    def action_shape(self):
        return self._data["action_shape"]

    def get_val_batch(self):
    #     imgs = self._images[self.num_train:self.num_train+self.num_val]
    #     transitions = self._transitions[self.num_train:self.num_train+self.num_val]
        return self.val_imgs, None, self.val_actions


if __name__ == '__main__':
    pass