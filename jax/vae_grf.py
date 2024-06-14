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
        self.logrange_prior = jnp.array(jnp.log(0.5))
        self.logvar_prior = jnp.array(jnp.log(1.))

    def corr_fct(self, array: Array) -> Float:
        if self.corr_type == "exp":
            return self.corr_exp(array)
        if self.corr_type == "m32":
            return self.corr_matern_32(array)
        raise ValueError("Unknown correlation type")

    def minus_kld(self, mu: Array, logvar: Array) -> Array:
        covar_base_prior = self.corr_fct(
            self.array_of_E_dist
        )[None]
        inv_covar_base_prior = self.get_base_invert(covar_base_prior)

        mu_col = mu.reshape(mu.shape[0], -1)
        invSigma_times_mu_col = self.get_matrix_vector_product(
            inv_covar_base_prior,
            mu
        ).reshape(mu.shape[0], -1)

        log_det_covar = self.get_logdeterminant_base(covar_base_prior).squeeze()

        # Tr(\Sigma^-1 m m.T)
        trace_1 = jnp.sum(invSigma_times_mu_col * mu_col, axis=-1)

        # Tr(\Sigma^-1 * diag(\sigma1,...,\sigmaN))
        trace_2 = jnp.sum(inv_covar_base_prior[:, 0, 0] * jnp.exp(logvar),
                axis=(1, 2))

        # log det L
        log_det_L = jnp.sum(logvar, axis=(1, 2))


        # To check the equivalency with the vanilla VAE
        # if logrange_prior = log(0.001) and logvar_prior = log(1)
        #jax.debug.print("trace_1 {x}", x=(trace_1[0], jnp.sum(mu[0]**2)))
        #jax.debug.print("trace_2 {x}", x=(trace_2[0],
        #    jnp.sum(jnp.exp(logvar[0]))))
        #jax.debug.print("logdet covar {x}", x=(log_det_covar, 0))
        #jax.debug.print("logdet L {x}", x=(log_det_L[0],
        #    jnp.sum(logvar[0])))

        # N
        N = self.latent_img_size ** 2

        # average over the latent_size**2 to be equal to KLD in VAE
        return 0.5*1/N*jnp.mean(- log_det_covar - trace_1 - trace_2 + N + log_det_L)

    def log_likelihood_grf(self, x):
        """
        compute the log density of a N-2D multivariate gaussian distributions
        x is of dimensions (N, latent_img_size, latent_img_size)
        """
        covar_base_prior = self.corr_fct(
            self.array_of_E_dist
        )[None]
        inv_covar_base_prior = self.get_base_invert(covar_base_prior)

        mu = jnp.mean(x, axis=(1, 2))
        x_norm_col = (x - mu[:, None, None]).reshape(x.shape[0], -1)

        invSigma_times_x_norm_col = self.get_matrix_vector_product(
            inv_covar_base_prior,
            x_norm_col,
        ).reshape(x_norm_col.shape[0], -1)
        # Tr(\Sigma^-1 m m.T)
        trace = jnp.sum(invSigma_times_x_norm_col * x_norm_col, axis=-1)

        log_det_covar = self.get_logdeterminant_base(covar_base_prior).squeeze()

        N = self.latent_img_size ** 2

        # average over the latent dimensions
        return 0.5 * 1 / N * jnp.mean(-N * jnp.log(2 * jnp.pi) - log_det_covar
                - trace)

    def corr_exp(self, a: Array) -> Float:
        """
        Return the value of the exponential correlation function for all
        distances given in a at the current parameter value of self.logrange_prior and
        self.logvar_prior
        """
        range_ = jnp.exp(self.logrange_prior)
        var_ = jnp.exp(self.logvar_prior)
        return var_ * jnp.exp(-a / range_)

    def corr_matern_32(self, a: Array) -> Float:
        """
        Return the value of the matern 3/2 correlation function for all
        distances given in a at the current parameter value of self.logrange_prior and
        self.logvar_prior
        """
        range_ = jnp.exp(self.logrange_prior)
        var_ = jnp.exp(self.logvar_prior)
        return var_ * (a / range_ + 1) * jnp.exp(-a / range_)

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
        lx, ly = b.shape[-1], b.shape[-2]
        return jnp.sqrt(lx * ly) * jnp.real(
                    jnp.fft.fft2(
                        jnp.fft.fft2(b, norm="ortho") *
                        jnp.fft.ifft2(v, norm="ortho"),
                    norm="ortho")
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
                (B + 1e-6) ** (-1),
                norm='ortho'
            )
        )
        return res

    def get_logdeterminant_base(self, covar_bases):
        '''
        Expects a [Channels, W, H], everything is vectorized over the
        first two channels
        '''
        lx, ly = covar_bases.shape[-1], covar_bases.shape[-2]
        B = jnp.sqrt(lx * ly) * jnp.fft.fft2(covar_bases, norm="ortho")
        logdet = jnp.sum(jnp.log(jnp.real(B) + 1e-6), axis=(-2, -1))
        return logdet
