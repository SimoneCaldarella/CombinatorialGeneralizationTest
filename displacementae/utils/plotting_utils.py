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
# @title          :displacementae/utils/plotting_utils.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :22/11/2021
# @version        :1.0
# @python_version :3.7.4

import matplotlib.pyplot as plt
import numpy as np
import torch
from argparse import Namespace

import utils.misc as misc
from sklearn.random_projection import GaussianRandomProjection


_DEFAULT_PLOT_CONFIG = [12, 5, 8] # fontsize, linewidth, markersize

# Simple 2D regression grid.
_TWO_D_MISC = Namespace()
_TWO_D_MISC.x_range = [-5, 5]
_TWO_D_MISC.y_range = [-5, 5]
_TWO_D_MISC.x_range_medium = [-1.2, 1.2]
_TWO_D_MISC.y_range_medium = [-1.2, 1.2]
_TWO_D_MISC.x_range_narrow = [-0.3, 0.3]
_TWO_D_MISC.y_range_narrow = [-0.3, 0.3]


def plot_reconstruction(dhandler, nets, config, device, logger, 
                        figname):
    if config.plot_on_black:
        plt.style.use('dark_background')
        
    imgs, latents, dj = dhandler.get_val_batch()
    X1 = torch.FloatTensor(imgs[:,0]).to(device)
    if config.intervene:
        X2 = torch.FloatTensor(imgs[:,1]).to(device)
    else:
        X2 = X1.clone()
    dj = torch.FloatTensor(dj).to(device).squeeze()
    if config.intervene:
        h, mu, logvar = nets(X2, dj[:, dhandler.varied_in_action])
    else:
        h, mu, logvar = nets(X2, None)
    X2_hat = torch.sigmoid(h)
    nrows = 7
    ncols = 3
    fig, axes = plt.subplots(nrows,ncols,figsize=(5,8))
    kwargs={'vmin':0,'vmax':1,'cmap':'gray'}
    for row in range(nrows):
        axes[row,0].imshow(X1[row,0].cpu().numpy(),**kwargs)
        axes[row,1].imshow(X2[row,0].cpu().numpy(),**kwargs)
        axes[row,2].imshow(X2_hat[row,0].cpu().numpy(),**kwargs)
        if config.plot_on_black:
            for i in range(3):
                axes[row,i].axes.xaxis.set_visible(False)
                axes[row,i].axes.yaxis.set_visible(False)
        else:
            for i in range(3):
                axes[row,i].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0.1)


    if figname is not None:
        figname += 'reconstructions.pdf'
        plt.savefig(figname,bbox_inches='tight')
        logger.info(f'Figure saved {figname}')
    plt.close(fig)
    

def plot_manifold(dhandler, nets, shared, config, device, logger, mode,
                epoch, vary_latents=[3], plot_latent=[0,1], figname=None):
    """
    Produces colored scatter plot of the latent representation of 
    the different positions in the joint space.

    A 1D or 2D grid of joint positions are generated, 
    corresponding images are forwarded through the encoder.
    Resulting latent units are 
    """
    n_repr_units = nets.n_repr_units
    if max(plot_latent) >= n_repr_units:
        raise ValueError(
            "Requested plotting a representational unit which index: "+
            f"{max(plot_latent)} is too large for the "+
            f"number of representational units: {n_repr_units}")
    ts, lw, ms = _DEFAULT_PLOT_CONFIG
    if config.plot_on_black:
        plt.style.use('dark_background')

    indices = dhandler.get_indices_vary_latents(vary_latents)
    latents = dhandler.latents[indices][:,vary_latents]
    batch_size = config.batch_size
    n_batches = len(indices) // batch_size + 1
    
    results = []

    for i in range(n_batches):
        batch_indices = indices[ i * batch_size : (i+1) * batch_size]
        images = dhandler.images[batch_indices]
        X = torch.FloatTensor(images).to(device)
        with torch.no_grad():
            h, mu, logvar = nets.encode(X)
            results.append(h[:,plot_latent].cpu().numpy())
    results = np.vstack(results).squeeze()

    if config.plot_on_black:
        kwargs={'cmap':'summer'}
    else:
        kwargs={}

    for i in range(len(vary_latents)):
        latent = vary_latents[i]
        latent_name = dhandler.get_latent_name(latent)
        fig, ax = plt.subplots(figsize=(8,7))
        if len(plot_latent) == 1:
            f = ax.scatter(x=latents[:,i], y=results)
            ax.set_xlabel(f'true label {latent}', fontsize=ts)
            ax.set_ylabel(f'latent {plot_latent[0]}', fontsize=ts)
        if len(plot_latent) == 2:
            f = ax.scatter(x=results[:,0], y=results[:,1], c=latents[:,i],
                            **kwargs)
            ax.set_xlabel(f'latent {plot_latent[0]}', fontsize=ts)
            ax.set_ylabel(f'latent {plot_latent[1]}', fontsize=ts)
            dx = np.abs(results).max()
            if config.spherical:
                ax.set_xlim(_TWO_D_MISC.x_range_medium)
                ax.set_ylim(_TWO_D_MISC.y_range_medium)
            if dx <= 0.3:
                ax.set_xlim(_TWO_D_MISC.x_range_narrow)
                ax.set_ylim(_TWO_D_MISC.y_range_narrow)
            else:
                ax.set_xlim(_TWO_D_MISC.x_range)
                ax.set_ylim(_TWO_D_MISC.y_range)
            plt.colorbar(f)
        ax.set_title('Manifold latent for latent: ' + latent_name)
        if figname is not None:
            figname1 = figname + 'repr_manifold_latent=' + misc.ints_to_str(plot_latent) 
            figname1 += '_true='+ misc.ints_to_str(latent) + '.pdf'
            plt.savefig(figname1)
            logger.info(f'Figure saved {figname1}')
        plt.close(fig)

def plot_manifold_pca(dhandler, nets, shared, config, device, logger, mode,
                epoch, vary_latents=[3], figname=None):
    """
    Produces colored scatter plot of the latent representation of 
    the different positions in the joint space.

    A 1D or 2D grid of joint positions are generated, 
    corresponding images are forwarded through the encoder.
    Resulting latent units are 
    """
    n_repr_units = nets.n_repr_units
    ts, lw, ms = _DEFAULT_PLOT_CONFIG
    if config.plot_on_black:
        plt.style.use('dark_background')

    indices = dhandler.get_indices_vary_latents(vary_latents)
    latents = dhandler.latents[indices][:,vary_latents]
    batch_size = config.batch_size
    n_batches = len(indices) // batch_size + 1
    
    results = []

    for i in range(n_batches):
        batch_indices = indices[ i * batch_size : (i+1) * batch_size]
        images = dhandler.images[batch_indices]
        X = torch.FloatTensor(images).to(device)
        with torch.no_grad():
            h, mu, logvar = nets.encode(X)
            results.append(h[:,:].cpu().numpy())
    results = np.vstack(results).squeeze()

    # PCA Projection
    pca = GaussianRandomProjection(n_components=2)
    latent2d = pca.fit_transform(results)


    if config.plot_on_black:
        kwargs={'cmap':'summer'}
    else:
        kwargs={}

    for i in range(len(vary_latents)):
        latent = vary_latents[i]
        latent_name = dhandler.get_latent_name(latent)
        fig, ax = plt.subplots(figsize=(8,7))

        f = ax.scatter(x=latent2d[:,0], y=latent2d[:,1], c=latents[:,i],
                        **kwargs)
        ax.set_xlabel(f'latent component 0', fontsize=ts)
        ax.set_ylabel(f'latent component 1', fontsize=ts)
        dx = np.abs(latent2d).max()
        if config.spherical:
            ax.set_xlim(_TWO_D_MISC.x_range_medium)
            ax.set_ylim(_TWO_D_MISC.y_range_medium)
        if dx <= 0.3:
            ax.set_xlim(_TWO_D_MISC.x_range_narrow)
            ax.set_ylim(_TWO_D_MISC.y_range_narrow)
        else:
            ax.set_xlim(_TWO_D_MISC.x_range)
            ax.set_ylim(_TWO_D_MISC.y_range)
        
        plt.colorbar(f)
        
        ax.set_title('Manifold latent for latent: ' + latent_name)
        if figname is not None:
            figname1 = figname + 'repr_manifold_pca' 
            figname1 += '_true='+ misc.ints_to_str(latent) + '.pdf'
            plt.savefig(figname1)
            logger.info(f'Figure saved {figname1}')
        plt.close(fig)




T_SERIES = ["bce_loss","kl_loss","learned_alpha"]
def plot_curves(shared,config,logger,figname=None,val_name=None):
    if config.plot_on_black:
        plt.style.use('dark_background')
    epochs = np.arange(len(vars(shared)[T_SERIES[0]]))*config.val_epoch
    for key in T_SERIES:
        if key in shared:
            fig, ax = plt.subplots(figsize=(8,7))
            ax.plot(epochs,vars(shared)[key])
            ax.set_xlabel('epochs')
            ax.set_ylabel(key)
            if figname is not None:
                figname1 = figname + 'curve_' + key + '.pdf'
                plt.savefig(figname1)
            plt.close()
