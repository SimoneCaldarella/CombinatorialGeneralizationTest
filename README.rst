========================================================================
Combinatorial Generalization with Homomorphism Autoencoder
========================================================================

Testing Homomorphism Autoencoder by Keurti et al. on Combinatorial Generalization benchmarks by Montero et al. for the final project of the course "Advanced Topics in Machine Learning and Optimization"


No recombination

```
cd displacementae/homomorphism/
python3 train_block_mlp_repr.py --dataset=dsprites --data_root=[BASEPATH]/dsprites-dataset --cyclic_trans --fixed_in_intervention=0,1,2,3 --fixed_in_sampling=0,1,2,3 --fixed_values=0,1,5,14 --distrib=uniform --displacement_range=-10,10 --n_steps=2 --rotate_actions=45 --num_train=10000 --batch_size=500 --epochs=101 --log_wandb --lr=0.001 --toggle_training_every=2,2 --shuffle=1 --use_adam --use_cuda --conv_channels=64,64,64,64 --kernel_sizes=6,4,4,4 --strides=2,2,1,1 --lin_channels=1024 --net_act=relu --dims=2,2 --group_hidden_units=128,128 --reconstruct_first --exponential_map --latent_loss --latent_loss_weight=400 --val_epoch=10 --num_val=500 --plot_epoch=10 --plot_manifold_latent=[0,1] --plot_manifold --plot_reconstruction --plot_pca --plot_vary_latents=[4,5]
```

Recombination to Element

```
cd displacementae/homomorphism/
python3 train_block_mlp_repr.py --combinatorial_indices=[BASEPATH]/remove_from_train_shape_equal_1__scale_equal_5__orientation_equal_14__posX_equal_15__posY_greater_equal_15.json --dataset=dsprites --data_root=[BASEPATH]/dsprites-dataset --cyclic_trans --fixed_in_intervention=0,1,2,3 --fixed_in_sampling=0,1,2,3 --fixed_values=0,1,5,14 --distrib=uniform --displacement_range=-10,10 --n_steps=2 --rotate_actions=45 --num_train=10000 --batch_size=500 --epochs=101 --log_wandb --lr=0.001 --toggle_training_every=2,2 --shuffle=1 --use_adam --use_cuda --conv_channels=64,64,64,64 --kernel_sizes=6,4,4,4 --strides=2,2,1,1 --lin_channels=1024 --net_act=relu --dims=2,2 --group_hidden_units=128,128 --reconstruct_first_only --exponential_map --latent_loss --latent_loss_weight=400 --val_epoch=10 --num_val=500 --plot_epoch=10 --plot_manifold_latent=[0,1] --plot_manifold --plot_reconstruction --plot_pca --plot_vary_latents=[4,5]
```

Recombination to Range

```
cd displacementae/homomorphism/
python3 train_block_mlp_repr.py --combinatorial_indices=[BASEPATH]/remove_from_train_shape_equal_1___posX_greater_equal_15__posY_greater_equal_15.json --dataset=dsprites --data_root=[BASEPATH]/dsprites-dataset --cyclic_trans --fixed_in_intervention=0,1,2,3 --fixed_in_sampling=0,1,2,3 --fixed_values=0,1,5,14 --distrib=uniform --displacement_range=-10,10 --n_steps=2 --rotate_actions=45 --num_train=10000 --batch_size=500 --epochs=101 --log_wandb --lr=0.001 --toggle_training_every=2,2 --shuffle=1 --use_adam --use_cuda --conv_channels=64,64,64,64 --kernel_sizes=6,4,4,4 --strides=2,2,1,1 --lin_channels=1024 --net_act=relu --dims=2,2 --group_hidden_units=128,128 --reconstruct_first_only --exponential_map --latent_loss --latent_loss_weight=400 --val_epoch=10 --num_val=500 --plot_epoch=10 --plot_manifold_latent=[0,1] --plot_manifold --plot_reconstruction --plot_pca --plot_vary_latents=[4,5]
```

Extrapolation

```
cd displacementae/homomorphism/
python3 train_block_mlp_repr.py --combinatorial_indices=[BASEPATH]/remove_from_train_shape_equal_1__posX_greater_equal_15.json --dataset=dsprites --data_root=[BASEPATH]/dsprites-dataset --cyclic_trans --fixed_in_intervention=0,1,2,3 --fixed_in_sampling=0,1,2,3 --fixed_values=0,1,5,14 --distrib=uniform --displacement_range=-10,10 --n_steps=2 --rotate_actions=45 --num_train=10000 --batch_size=500 --epochs=101 --log_wandb --lr=0.001 --toggle_training_every=2,2 --shuffle=1 --use_adam --use_cuda --conv_channels=64,64,64,64 --kernel_sizes=6,4,4,4 --strides=2,2,1,1 --lin_channels=1024 --net_act=relu --dims=2,2 --group_hidden_units=128,128 --reconstruct_first --exponential_map --latent_loss --latent_loss_weight=400 --val_epoch=10 --num_val=500 --plot_epoch=10 --plot_manifold_latent=[0,1] --plot_manifold --plot_reconstruction --plot_pca --plot_vary_latents=[4,5]
```
