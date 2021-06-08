#!/bin/sh

mpiexec -n 1 --allow-run-as-root python jukebox/train.py --hps=small_vqvae --name=small_vqvae --sample_length=262144 --bs=2 --nworkers=1 --audio_files_dir=HIPHOP --labels=False --train --aug_shift --aug_blend
mpiexec -n 1 --allow-run-as-root python jukebox/train.py --hps=small_vqvae,small_prior,all_fp16,cpu_ema --name=small_prior --sample_length=2097152 --bs=2 --nworkers=1 --audio_files_dir=HIPHOP --labels=False --train --test --aug_shift --aug_blend --restore_vqvae=logs/small_vqvae/checkpoint_latest.pth.tar --prior --levels=2 --level=1 --weight_decay=0.01 --save_iters=1000
mpiexec -n 1 --allow-run-as-root python jukebox/train.py --hps=small_vqvae,small_upsampler,all_fp16,cpu_ema --name=small_upsampler --sample_length 262144 --bs 2 --nworkers 1 --audio_files_dir HIPHOP --labels False --train --test --aug_shift --aug_blend --restore_vqvae logs/small_vqvae/checkpoint_latest.pth.tar --prior --levels 2 --level 0 --weight_decay 0.01 --save_iters 1000

