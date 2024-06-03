import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Int, Key, Bool, Array
from resnet import resnet18, ResNet

class VAE(eqx.Module):
    resnet: ResNet

    #conv1: eqx.nn.Conv2d
    #conv2: eqx.nn.Conv2d
    #conv3: eqx.nn.Conv2d
    #conv4: eqx.nn.Conv2d

    upconv1: eqx.nn.ConvTranspose2d
    upconv2: eqx.nn.ConvTranspose2d
    upconv3: eqx.nn.ConvTranspose2d
    upconv4: eqx.nn.ConvTranspose2d
    
    #norm_1: eqx.nn.BatchNorm
    #norm_2: eqx.nn.BatchNorm
    #norm_3: eqx.nn.BatchNorm

    norm1: eqx.nn.BatchNorm
    norm2: eqx.nn.BatchNorm
    norm3: eqx.nn.BatchNorm

    z_dim: Int
    
    def __init__(self, z_dim: Int, key: Key):
        self.z_dim = z_dim
        self.resnet = resnet18(
            key=key
        )
        keys = jax.random.split(key, 5)
        #self.conv1 = eqx.nn.Conv2d(in_channels=3, out_channels=64,
        #    kernel_size=7, stride=2, padding=3, key=keys[0])
        #self.conv2 = eqx.nn.Conv2d(in_channels=64, out_channels=64,
        #    kernel_size=4, stride=2, padding=1, key=keys[1])
        #self.conv3 = eqx.nn.Conv2d(in_channels=64, out_channels=128,
        #    kernel_size=4, stride=2, padding=1, key=keys[2])
        #self.conv4 = eqx.nn.Conv2d(in_channels=128, out_channels=2 * self.z_dim,
        #    kernel_size=1, stride=1, padding=0, key=keys[3])
        keys = jax.random.split(keys[4], 4)
        self.upconv1 = eqx.nn.ConvTranspose2d(in_channels=self.z_dim,
                out_channels=128,
            kernel_size=1, stride=1, padding=0, key=keys[0])
        self.upconv2 = eqx.nn.ConvTranspose2d(in_channels=128, out_channels=64,
            kernel_size=4, stride=2, padding=1, key=keys[1])
        self.upconv3 = eqx.nn.ConvTranspose2d(in_channels=64, out_channels=32,
            kernel_size=4, stride=2, padding=1, key=keys[2])
        self.upconv4 = eqx.nn.ConvTranspose2d(in_channels=32, out_channels=3,
            kernel_size=4, stride=2, padding=1, key=keys[3])

        #self.norm_1 = eqx.nn.BatchNorm(input_size=64, axis_name="batch")
        #self.norm_2 = eqx.nn.BatchNorm(input_size=64, axis_name="batch")
        #self.norm_3 = eqx.nn.BatchNorm(input_size=128, axis_name="batch")

        self.norm1 = eqx.nn.BatchNorm(input_size=128, axis_name="batch")
        self.norm2 = eqx.nn.BatchNorm(input_size=64, axis_name="batch")
        self.norm3 = eqx.nn.BatchNorm(input_size=32, axis_name="batch")
        

    def encoder(self, x: Array, state: Array, key: Key) -> tuple[Array, Array]:
        z, state = self.resnet(x, state, key)
        #x = self.conv1(x)
        #x, state = self.norm_1(x, state)
        #x = jax.nn.relu(x)
        #x = self.conv2(x)
        #x, state = self.norm_2(x, state)
        #x = jax.nn.relu(x)
        #x = self.conv3(x)
        #x, state = self.norm_3(x, state)
        #x = jax.nn.relu(x)
        #z = self.conv4(x)
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

    def reparametrize(
        self, mu: Array, logvar: Array, key: Key, train: Bool, n_mcmc: Int=50
    ) -> Array:
        if train:
            std = jnp.exp(0.5 * logvar)
            eps = jax.random.normal(key, shape=(n_mcmc,) + std.shape)
            print(eps.shape)
            return eps * std[None] + mu[None]
        else:
            return mu[None]

    def __call__(
        self, x: Array, state : Array, key: Key, train: Bool
    ) -> tuple[Array, Array]:
        z, state = self.encoder(x, state, key)
        mu, logvar = z[:self.z_dim], z[self.z_dim:]
        z_samples = self.reparametrize(mu, logvar, key, train)

        def scan_fun(carry, z_sample):
            key = carry[0]
            key, subkey = jax.random.split(key, 2)
            x_rec, state = self.decoder(z_sample, carry[1], key)
            return (key, state), x_rec

        (_, state), x_rec = jax.lax.scan(
            scan_fun,
            (key, state),
            z_samples
        )
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
        else:
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

    def minus_kld(self, mu, logvar):
        return 0.5 * jnp.mean(
                1 + logvar - mu ** 2 - jnp.exp(logvar),
        )

    def elbo(self, x_rec, x, mu, logvar, beta):
        _, rec_terms = jax.lax.scan(
            lambda _, x_rec: (None, self.xent_continuous_ber(x_rec, x)),
            None,
            x_rec
        )
        rec_term = jnp.mean(rec_terms)
        #rec_term = -jnp.mean((x_rec-x)**2)#self.xent_continuous_ber(x_rec, x)
        #minus_kld = 0
        minus_kld = self.minus_kld(mu, logvar)

        L = (rec_term + beta * minus_kld)

        loss = L

        return loss, rec_term, -minus_kld
