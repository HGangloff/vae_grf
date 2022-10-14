# Variational Autoencoder with Gaussian Random Field prior

Repository linked with the publication

> Variational Autoencoder with Gaussian Random Field prior: application to
> unsupervised animal detection in aerial images, H. Gangloff, M.-T. Pham, L.
> Courtrai, S. Lefèvre, 2022.
> (https://hal.archives-ouvertes.fr/view/index/docid/3774853)

The model can be directly tested on the Livestock dataset which is provided to
reproduce the results from this section of the article.

To train a model run the file: `sh vae_train.sh`. For the classical VAE model,
set `corr_type=corr_id`, for the VAE-GRF model set `corr_type=corr_exp` or
`corr_type=corr_m32`. Dataset available is `livestock` for now.

To test a model run the file: `sh vae_test.sh` with appropriate parameters.
Some checkpoints files are provided in `torch_checkpoints` to reproduce
directly the results from the article.

The code is built with PyTorch and other standard librairies.

For more details, refer to the publication.
