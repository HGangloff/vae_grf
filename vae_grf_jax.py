import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Int, Bool, Array, Float, Key

from vae_jax import VAE

class VAE_GRF(VAE):
    corr_type: str = eqx.field(static=True)
    array_of_E_dist: Array = eqx.field(init=False)
    logrange_prior: Float = eqx.field(init=False)
    logvar_prior: Float = eqx.field(init=False)

    def __post_init__(self, init_key: Key):
        """
        Note that init_key is an dataclasses.InitVar from super and it must be
        in __post_init__ of this child class too
        """
        super().__post_init__(init_key) # this must be explicitely done,
        # otherwise all the __post_init__ attributes of super() are not
        # instanciate and this raises an error
        self.array_of_E_dist = self.get_euclidean_dist_torus_array(self.latent_img_size,
                self.latent_img_size)
        self.logrange_prior = jnp.array(0.)
        self.logvar_prior = jnp.array(0.)

    def corr_fct(self, array: Array) -> Float:
        if self.corr_type == "exp":
            return self.corr_exp(array)
        raise ValueError("Unknown correlation type")

    def minus_kld(self, mu: Array, logvar: Array) -> Array:
        print(mu.shape, logvar.shape)
        covar_base_prior = self.corr_fct(
            self.array_of_E_dist
        )[None]
        print(covar_base_prior.shape)
        inv_covar_base_prior = self.get_base_invert(covar_base_prior)
        print(inv_covar_base_prior.shape)

        #mu_col = jnp.transpose(mu, axes=(0, 2, 1)).reshape(mu.shape[0], -1)
        #print(mu_col.shape)
        invSigma_times_mu_col = self.get_matrix_vector_product(
            inv_covar_base_prior,
            mu # NOTE check the order !
        ).transpose((0, 2, 1)).reshape(mu.shape[0], -1, 1)
        print(invSigma_times_mu_col.shape)

        log_det_covar = self.get_logdeterminant_base(covar_base_prior)
        print(log_det_covar.shape)

        #return 0.5 * jnp.mean(
        #    ##### E_{q(z|x)}[p(x|z)] #####
        #    # log det \Sigma
        #    -self.get_logdeterminant_base(covar_base_prior).squeeze() -

        #    # Tr(\Sigma^-1 m m.T)
        #    torch.sum(
        #        torch.mul(
        #            invSigma_times_mu_col,
        #            mu_col[..., None]
        #        ),
        #        dim=(-1, -2)
        #    ).squeeze() -

        #    # Tr(\Sigma^-1 * diag(\sigma1,...,\sigmaN)) # model A2
        #    torch.sum(
        #        torch.mul(
        #            #inv_covar_base_prior[:, :, 0, 0],
        #            inv_covar_base_prior[:, 0, 0, 0][:, None, None, None],
        #            var_
        #        ),
        #        dim=(-1, -2)
        #    ).squeeze() +
        #    #####  E_{q(z|x)}[q(z|x)] #####
        #    # log det L
        #    torch.sum(self.logvar, dim=(-1, -2)) +
        #    # N
        #    self.latent_img_size ** 2,
        #    dim=1
        #)

    def corr_exp(self, a: Array) -> Float:
        """
        Return the value of the exponential correlation function for all
        distances given in a at the current parameter value of self.logrange_prior and
        self.logvar_prior
        """
        range_ = jnp.exp(self.logrange_prior)
        var_ = jnp.exp(self.logvar_prior)
        return var_[None] * jnp.exp(-a / range_[None])

    def euclidean_dist_torus(self, x1: Array, x2: Array, lx: Int, ly: Int) -> Float:
        """
        Compute the Euclidean distance on the torus obtained from the regular
        lattice of size (self.latent_img_size * self.latent_img_size)
        """
        return jnp.sqrt(jnp.amin(jnp.array([jnp.abs(x1[0] - x2[0]), lx -
            jnp.abs(x1[0] - x2[0])])) ** 2 + jnp.amin(jnp.array([jnp.abs(x1[1] - x2[1]),
                ly - jnp.abs(x1[1] - x2[1])])) ** 2)


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
                        jnp.fft.fft2(b, norm="ortho") *
                        jnp.fft.ifft2(v, norm="ortho")
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
        Expects a [Channels, W, H], everything is vectorized over the
        first two channels
        '''
        B = jnp.fft.fft2(covar_bases) # default = no normalization : OK!
        logdet = jnp.sum(jnp.log(jnp.real(B)), axis=(-2, -1))
        return logdet
