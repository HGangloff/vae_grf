#!/bin/bash

python3 vae_test.py\
    --exp=VAEGRF_livestock_corrid\
    --dataset=livestock\
    --lr=1e-4\
    --img_size=256\
    --batch_size=16\
    --batch_size_test=8\
    --latent_img_size=32\
    --z_dim=256\
    --beta=1\
    --nb_channels=3\
    --model=vae_grf\
    --corr_type=corr_id\
    --params_id=4000\
