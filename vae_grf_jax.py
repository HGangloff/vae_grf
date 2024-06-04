import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Int, Key, Bool, Array, Float

from vae_jax import VAE

class VAE_GRF(VAE):
    #corr_fct: str = eqx.field(static=True, default_factory=lambda:"exp")
    logrange_prior: Float = 0.
    logvar_prior: Float = 0.

    #def __init__(self, img_size, latent_img_size, z_dim, corr_fct, key):
    #    self.logrange_prior = jnp.array(0.)
    #    self.logvar_prior = jnp.array(0.)

    #def __post_init__(self):
    #    if self.corr_fct == "exp":
    #        self.corr_fct = self.corr_exp

    #def minus_kld(self, mu: Array, logvar: Array) -> Array:

    #    covar_base_prior = self.corr_fct(
    #        array='eucli_dist_array_latent_size',
    #        logrange=torch.log(range_),
    #        logsigma=self.logsigma_prior.detach()
    #    )[:, None]
    #    inv_covar_base_prior = self.get_base_invert(covar_base_prior)

    #    invSigma_times_mu_col = self.get_matrix_vector_product(
    #        inv_covar_base_prior,
    #        mu_
    #    ).transpose(-1, -2).reshape(mu_.shape[0], mu_.shape[1], -1, 1)

    #def corr_exp(self, x1: Array, x2: Array) -> Float:
    #    """
    #    Return the value of the exponential correlation function between
    #    points x1 and x2 at the current parameter value of self.logrange_prior and
    #    self.logvar_prior
    #    """
    #    range_ = jnp.exp(self.logrange_prior)
    #    var_ = jnp.exp(self.logvar_prior)
    #    return var_ * jnp.exp(-self.euclidean_dist_torus(x1, x2) / range_)

    def euclidean_dist_torus(self, x1: Array, x2: Array, lx: Int, ly: Int) -> Float:
        """
        Compute the Euclidean distance on the torus obtained from the regular
        lattice of size (self.latent_img_size * self.latent_img_size)
        """
        return jnp.sqrt(jnp.amin(jnp.abs(x1[0] - x2[0]), lx -
            jnp.abs(x1[0] - x2[0])) ** 2 + jnp.amin(jnp.abs(x1[1] - x2[1]),
                ly - jnp.abs(x1[1] - x2[1])) ** 2)


    def get_euclidean_dist_torus_array(self, lx, ly):
        '''
        precompute the distance array on a torus formed by the regular lattice
        of size lx ly
        '''
        zz = jnp.stack(jnp.meshgrid(jnp.arange(lx), jnp.arange(ly)), axis=-1)
        distances_to_00 = jax.vmap(
            jax.vmap(
                lambda z : self.euclidean_dist_torus(z, jnp.array([0, 0]), lx,
                    ly),
                0, 0
            ),
            0, 0
        )(zz)
        distances_to_0ly = jax.vmap(
            jax.vmap(
                lambda z : self.euclidean_dist_torus(z, jnp.array([0, ly]), lx,
                    ly),
                0, 0
            ),
            0, 0
        )(zz)
        distances_to_lx0 = jax.vmap(
            jax.vmap(
                lambda z : self.euclidean_dist_torus(z, jnp.array([lx, 0]), lx,
                    ly),
                0, 0
            ),
            0, 0
        )(zz)
        distances_to_lxly = jax.vmap(
            jax.vmap(
                lambda z: self.euclidean_dist_torus(z, jnp.array([lx, ly]), lx,
                    ly),
                0, 0
            ),
            0, 0
        )(zz)

        return jnp.min(jnp.stack([distances_to_00, distances_to_0ly,
            distances_to_lx0, distances_to_lxly], axis=0), axis=0)


    def get_matrix_vector_product(self, b, v):
        '''
        b is the base of a block circulant matrix of size lx*ly x lx * ly
        v is the lx * ly 1D vector
        return a lx * ly 1D vector
        '''
        return jnp.real(
                    jnp.fft.fft2(
                        jnp.mul(
                            jnp.fft.fft2(b, norm="ortho"),
                            jnp.fft.ifft2(v, norm="ortho")
                            )
                    )
                )
    
    def get_base_invert(self, b):
        '''
        If b is the base of a matrix B, returns bi, the base of B^-1 with the
        direct formula sing Fourier space
        '''
        lx, ly = b.shape[-1], b.shape[-2]
        B = jnp.fft.fft2(b, norm='ortho')
        res = 1 / (lx * ly) * jnp.real(
            jnp.fft.ifft2(
                B ** (-1),
                norm='ortho'
            )
        )
        return res

    def get_logdeterminant_base(self, covar_bases):
        '''
        Expects a [Batch, Channels, W, H], everything is vectorized over the
        first two channels
        '''
        B = jnp.fft.fft2(covar_bases) # default = no normalization : OK!
        logdet = jnp.sum(jnp.log(jnp.real(B)), axis=(-2, -1))
        return logdet
