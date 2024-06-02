import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Int, Key, Bool, Array
from resnet import resnet18, ResNet

class VAE(eqx.Module):
    resnet: ResNet
    upconv1: eqx.nn.ConvTranspose2d
    upconv2: eqx.nn.ConvTranspose2d
    upconv3: eqx.nn.ConvTranspose2d
    upconv4: eqx.nn.ConvTranspose2d
    
    norm1: eqx.nn.BatchNorm
    norm2: eqx.nn.BatchNorm
    norm3: eqx.nn.BatchNorm

    z_dim: Int
    
    def __init__(self, z_dim: Int, key: Key):
        self.z_dim = z_dim
        self.resnet = resnet18(
            key=key
        )
        keys = jax.random.split(key, 3)
        self.upconv1 = eqx.nn.ConvTranspose2d(in_channels=self.z_dim, out_channels=128,
            kernel_size=1, stride=1, padding=0, key=keys[0])
        self.upconv2 = eqx.nn.ConvTranspose2d(in_channels=128, out_channels=64,
            kernel_size=4, stride=2, padding=1, key=keys[1])
        self.upconv3 = eqx.nn.ConvTranspose2d(in_channels=64, out_channels=32,
            kernel_size=4, stride=2, padding=1, key=keys[2])
        self.upconv4 = eqx.nn.ConvTranspose2d(in_channels=32, out_channels=3,
            kernel_size=4, stride=2, padding=1, key=keys[2])

        self.norm1 = eqx.nn.BatchNorm(input_size=128, axis_name="batch")
        self.norm2 = eqx.nn.BatchNorm(input_size=64, axis_name="batch")
        self.norm3 = eqx.nn.BatchNorm(input_size=32, axis_name="batch")
        

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
        return x, state

    def reparametrize(
        self, mu: Array, logvar: Array, key: Key, train: Bool
    ) -> Array:
        if train:
            std = jnp.exp(0.5 * logvar)
            eps = jax.random.normal(key, shape=std.shape)
            return eps * std + mu
        else:
            return mu

    def __call__(
        self, x: Array, state : Array, key: Key, train: Bool
    ) -> tuple[Array, Array]:
        z, state = self.encoder(x, state, key)
        mu, logvar = z[:self.z_dim], z[self.z_dim:]
        mu = self.reparametrize(mu, logvar, key, train)
        x_rec, state = self.decoder(mu, state, key)
        return x_rec, state

