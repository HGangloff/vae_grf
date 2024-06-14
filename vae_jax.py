import dataclasses
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Int, Key, Bool, Array, Float
from resnet import resnet18, ResNet

class VAE(eqx.Module):

    # __init__ args order is important! Those are required in __init__ call.
    # The automatic __init__() constructed by dataclasses just perform the
    # assignement
    img_size: Int
    latent_img_size: Int
    z_dim: Int
    init_key: dataclasses.InitVar[Key] # not a true field, there are just here to be passed
    # to __init__ and __post_init__
    
    # init=False otherwise they are required in __init__ call 
    resnet: ResNet = eqx.field(init=False)
    upconv1: eqx.nn.ConvTranspose2d = eqx.field(init=False)
    upconv2: eqx.nn.ConvTranspose2d = eqx.field(init=False)
    upconv3: eqx.nn.ConvTranspose2d = eqx.field(init=False)
    upconv4: eqx.nn.ConvTranspose2d = eqx.field(init=False)
    norm1: eqx.nn.BatchNorm = eqx.field(init=False)
    norm2: eqx.nn.BatchNorm = eqx.field(init=False)
    norm3: eqx.nn.BatchNorm = eqx.field(init=False)

    def __post_init__(self, init_key: Key):
        keys = jax.random.split(init_key, 5)
        self.resnet = resnet18(
            key=keys[4]
        )
        self.upconv1 = eqx.nn.ConvTranspose2d(in_channels=self.z_dim,
                out_channels=128,
            kernel_size=1, stride=1, padding=0, key=keys[0])
        self.upconv2 = eqx.nn.ConvTranspose2d(in_channels=128, out_channels=64,
            kernel_size=4, stride=2, padding=1, key=keys[1])
        self.upconv3 = eqx.nn.ConvTranspose2d(in_channels=64, out_channels=32,
            kernel_size=4, stride=2, padding=1, key=keys[2])
        self.upconv4 = eqx.nn.ConvTranspose2d(in_channels=32, out_channels=3,
            kernel_size=4, stride=2, padding=1, key=keys[3])

        self.norm1 = eqx.nn.BatchNorm(input_size=128, axis_name="batch", momentum=0.1)
        self.norm2 = eqx.nn.BatchNorm(input_size=64, axis_name="batch", momentum=0.1)
        self.norm3 = eqx.nn.BatchNorm(input_size=32, axis_name="batch", momentum=0.1)
 

    def encoder(self, x: Array, state: Array, key: Key) -> tuple[Array, Array]:
        z, state = self.resnet(x, state, key)
        return z, state

    def decoder(self, z: Array, state: Array, key: Key) -> tuple[Array, Array]:
        x = self.upconv1(z)
        x, state = self.norm1(x, state)
        x = jax.nn.relu(x)
        x = self.upconv2(x)
        x, state = self.norm2(x, state)
        x = jax.nn.relu(x)
        x = self.upconv3(x)
        x, state = self.norm3(x, state)
        x = jax.nn.relu(x)
        x = self.upconv4(x)
        x = jax.nn.sigmoid(x)
        return x, state

    def v_decoder(self, z, state, key):
        v_decoder = jax.vmap(self.decoder, (0, None, None), (0, None), axis_name="mcmc")
        x_rec, state = v_decoder(z, state, key)
        return x_rec, state

    def reparametrize(
        self, mu: Array, logvar: Array, key: Key, train: Bool, n_mcmc: Int=1
    ) -> Array:
        if train:
            std = jnp.exp(0.5 * logvar)
            eps = jax.random.normal(key, shape=(n_mcmc,) + std.shape)
            return eps * std[None] + mu[None]
        return mu[None]

    def __call__(
        self, x: Array, state : Array, key: Key, train: Bool
    ) -> tuple[Array, Array]:
        keys = jax.random.split(key, 3)
        z, state = self.encoder(x, state, keys[0])
        mu, logvar = z[:self.z_dim], z[self.z_dim:]
        z_samples = self.reparametrize(mu, logvar, keys[1], train)

        x_rec, state = self.decoder(z_samples.squeeze(), state, keys[2])
        return x_rec, state, mu, logvar

    def xent_continuous_ber(
        self, x_rec: Array, x: Array, pixelwise: Bool = False
    ) -> Array:
        ''' p(x_i|z_i) a continuous bernoulli '''
        eps = 1e-6
        def log_norm_const(x):
            # numerically stable computation
            x = jnp.clip(x, min=eps, max=1 - eps)
            x = jnp.where(
                jnp.logical_or(x < 0.49, x > 0.51),
                x,
                0.49 * jnp.ones_like(x)
            )
            return jnp.log((2 * jnp.arctanh(1 - 2 * x)) /
                            (1 - 2 * x) + eps)
        if pixelwise:
            return (x * jnp.log(x_rec + eps) +
                            (1 - x) * jnp.log(1 - x_rec + eps) +
                            log_norm_const(x_rec))
        return jnp.mean(
            jnp.sum(x * jnp.log(x_rec + eps) +
            (1 - x) * jnp.log(1 - x_rec + eps) +
            log_norm_const(x_rec), axis=0 # sum on the channels
        ))

    @classmethod
    def mean_from_lambda(self, l: Array) -> Array:
        ''' because the mean of a continuous bernoulli is not its lambda '''
        eps = 1e-6
        l = jnp.clip(l, min=eps, max=1 - eps)
        l = jnp.where(
            jnp.logical_or(l < 0.49, l > 0.51),
            l,
            0.49 * jnp.ones_like(l)
        )
        return l / (2 * l - 1) + 1 / (2 * jnp.arctanh(1 - 2 * l))

    def minus_kld(self, mu: Array, logvar: Array) -> Array:
        return 0.5 * jnp.mean(
                1 + logvar - mu ** 2 - jnp.exp(logvar),
        )

    def elbo(self, x_rec: Array, x: Array, mu: Array, logvar: Array,
            beta: Float) -> Array:
        rec_term = self.xent_continuous_ber(x_rec, x)
        minus_kld = self.minus_kld(mu, logvar)

        L = (rec_term + beta * minus_kld)

        loss = L

        return loss, rec_term, -minus_kld
