import math
import numpy as np
import torch
from torch import nn
from torchvision.models.resnet import resnet18 as resnet18
from torchvision.transforms.functional import rgb_to_grayscale
from vae import VAE

class VAE_GRF(VAE):

    def __init__(self, img_size, nb_channels, latent_img_size, z_dim,
        corr_type, batch_size, beta=1):
        '''
        '''
        super().__init__(img_size, nb_channels, latent_img_size, z_dim, beta)


        self.batch_size = batch_size
        self.corr_type = corr_type

        ## if we have p(z)=N(z,\mu,\Sigma)
        if self.corr_type == "corr_exp":
            self.corr_fct = self.corr_exp
        elif self.corr_type == "corr_gauss":
            self.corr_fct = self.corr_gauss
        elif self.corr_type == "corr_m32":
            self.corr_fct = self.corr_matern_32
        # if we have p(z)=N(z,0,1)
        elif self.corr_type == "corr_id":
            self.corr_fct = self.corr_identity
        else:
            raise RuntimeError("Unrecognized corr type")

        self.logrange_prior = nn.Parameter(
            torch.log(torch.Tensor([1]))
        )
        self.logsigma_prior = nn.Parameter(
            torch.log(torch.Tensor([0.1]))
        )
        #self.logsigma_prior.requires_grad=False
        self.register_buffer(
            'eucli_dist_array_latent_size',
            torch.empty(
                (self.latent_img_size, self.latent_img_size)
            )
        )
        self.set_euclidean_dist_torus_array(
            array='eucli_dist_array_latent_size',
        ) # Initialize !
        self.register_buffer(
            'eucli_dist_array_img_size',
            torch.empty(
                (self.img_size, self.img_size)
            )
        )
        self.set_euclidean_dist_torus_array(
            array='eucli_dist_array_img_size',
        ) # Initialize !

    def reparameterize(self, mu, logvar):
        '''
        Reparametrization trick inside fourier sampling
        '''
        if self.training:
            std = torch.exp(torch.mul(logvar, 0.5))
            eps = torch.randn_like(std)
            return eps * std + mu

        else:
            return mu


    def forward(self, x):

        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        self.mu = mu
        self.logvar = logvar

        return self.decoder(z), (mu, logvar)
       
    def kld(self):
        # NOTE -kld actually

        mu_ = self.mu
        mu_col = mu_.reshape(mu_.shape[0], mu_.shape[1], -1)
        var_ = self.logvar.exp()

        range_ = torch.exp(self.logrange_prior).detach()
        range_ /= 8
        covar_base_prior = self.corr_fct(
            array='eucli_dist_array_latent_size',
            logrange=torch.log(range_),
            logsigma=self.logsigma_prior.detach()
        )[None, None]
        inv_covar_base_prior = self.get_base_invert(covar_base_prior)

        invSigma_times_mu_col = self.get_matrix_vector_product(
            inv_covar_base_prior,
            mu_
        ).reshape(mu_.shape[0], mu_.shape[1], -1)

        return 0.5 * 1 / (self.latent_img_size ** 2) * torch.mean(
            # log det \Sigma
            ##### E_{q(z|x)}[p(x|z)] #####
            -self.get_logdeterminant_base(covar_base_prior).squeeze() -

            # Tr(\Sigma^-1 m m.T)
            torch.sum(
                invSigma_times_mu_col * mu_col,
                dim=(-1)
            ).squeeze() -

            # Tr(\Sigma^-1 * diag(\sigma1,...,\sigmaN)) # model A2
            torch.sum(
                inv_covar_base_prior[:, :, 0, 0] * var_,
                dim=(-1, -2)
            ).squeeze() +
            #####  E_{q(z|x)}[q(z|x)] #####
            # log det L
            torch.sum(self.logvar, dim=(-1, -2)) +
            # N
            self.latent_img_size ** 2,
            dim=1
        )

    def loglikelihood_prior(self, x):
        covar_base_prior = self.corr_fct(
            array='eucli_dist_array_img_size',
        )[None, None]
        inv_covar_base_prior = self.get_base_invert(covar_base_prior)

        #x = rgb_to_grayscale(x)

        sum_llkh_by_channels = 0
        for i in range(x.shape[1]):
            x_ = x[:, i, :, :]
            mu = torch.mean(x_, dim=(-1, -2))

            x_norm = x_ - mu[:, None, None]
            x_norm_col = x_norm.reshape(
                x.shape[0], -1
            )

            invSigma_times_x_norm_col = self.get_matrix_vector_product(
                inv_covar_base_prior,
                x_norm
            ).reshape(x_norm.shape[0], -1)

            sum_llkh_by_channels += 1/self.img_size**2 * torch.mean(- self.img_size ** 2 / 2 *
                    torch.log(torch.tensor(2 * math.pi))
                    - 0.5 *
                    self.get_logdeterminant_base(covar_base_prior).squeeze()
                    - 0.5 * torch.sum(
                        x_norm_col *
                        invSigma_times_x_norm_col, axis=(-1)
                        )
                    )
        return sum_llkh_by_channels

    def loss_function(self, recon_x, x):
        rec_term = self.xent_continuous_ber(recon_x, x)

        rec_term = torch.mean(rec_term)

        kld = torch.mean(self.kld())

        if self.corr_type == "corr_id":
            llkh_prior = 0
        else:
            llkh_prior = torch.mean(self.loglikelihood_prior(x))

        L = (rec_term + self.beta * kld) + llkh_prior

        loss = L

        loss_dict = {
            'loss': loss,
            'rec_term': rec_term,
            '-beta*kld': self.beta * kld,
            'llkh_prior':llkh_prior
        }

        return loss, loss_dict

    def loss_function_wrapper(self, x, err_type):
        (recon_mb), _ = self.forward(x)
        loss, loss_dict = self.loss_function(recon_mb, x)
        return loss_dict[err_type]

    def step(self, input_mb):
        x = input_mb


        (recon_mb), _ = self.forward(x)

        loss, loss_dict = self.loss_function(recon_mb, x)

        recon_mb = self.mean_from_lambda(recon_mb)

        return loss, recon_mb, loss_dict

    def set_euclidean_dist_torus_array(self, array):
        '''
        precompute the distance array
        '''
        b = self.state_dict()[array]

        lx, ly = b.shape[0], b.shape[1]
        z00 = torch.zeros((lx, 1, 2))
        zz = torch.stack([
            torch.stack([
                torch.full((ly, ), float(i)), torch.arange(ly)
            ], axis=1) for i in range(lx)
            ], axis=0
        )
        d1 = torch.squeeze(torch.cdist(z00, zz))
        z256256 = torch.stack([
            torch.Tensor([lx, ly])
            for i in range(lx)], axis=0
        )[:, None]
        d2 = torch.squeeze(torch.cdist(z256256, zz))
        z0256 = torch.stack([
            torch.Tensor([0, ly])
            for i in range(lx)], axis=0
        )[:, None]
        d3 = torch.squeeze(torch.cdist(z0256, zz))
        z2560 = torch.stack([
            torch.Tensor([lx, 0])
            for i in range(lx)], axis=0
        )[:, None]
        d4 = torch.squeeze(torch.cdist(z2560, zz))
        d, _ = torch.min(torch.stack([d1, d2, d3, d4], axis=0), dim=0)

        sd = self.state_dict()
        sd[array] = d
        self.load_state_dict(sd)

    def corr_exp(self, array, logsigma=None, logrange=None):
        if logrange is None:
            range_ = torch.exp(self.logrange_prior)
        else:
            range_ = torch.exp(logrange)

        if logsigma is None:
            sigma2_ = torch.pow(torch.exp(self.logsigma_prior), 2)
        else:
            sigma2_ = torch.pow(torch.exp(logsigma), 2)

        return (sigma2_ *
            torch.exp(-self.state_dict()[array] /
                range_))

    def corr_gauss(self, array, logsigma=None, logrange=None):
        if logrange is None:
            range_ = torch.exp(self.logrange_prior)
        else:
            range_ = torch.exp(logrange)

        if logsigma is None:
            sigma2_ = torch.pow(torch.exp(self.logsigma_prior), 2)
        else:
            sigma2_ = torch.pow(torch.exp(logsigma), 2)

        return (sigma2_ *
            torch.exp(-(self.state_dict()[array] /
                range_) ** 2))

    def corr_matern_32(self, array, logsigma=None, logrange=None):
        if logrange is None:
            range_ = torch.exp(self.logrange_prior)
        else:
            range_ = torch.exp(logrange)

        if logsigma is None:
            sigma2_ = torch.pow(torch.exp(self.logsigma_prior), 2)
        else:
            sigma2_ = torch.pow(torch.exp(logsigma), 2)

        return (sigma2_ *
                (self.state_dict()[array] / range_ + 1) *
                torch.exp(-self.state_dict()[array] / range_))

    def corr_identity(self, array, logsigma=None, logrange=None):
        '''
        To compare with the classic VAE is we want to
        '''
        temp = torch.zeros_like(
            self.corr_exp(
                array,
                logsigma=logsigma,
                logrange=logrange
            )
        )
        temp[:, 0, 0] = 1
        return temp
        
    ###########################
    # FOURIER FUNCTIONS BELOW #
    ###########################


    def get_matrix_vector_product(self, b, v):
        '''
        b is the base of a block circulant matrix of size lx*ly x lx * ly
        v is the lx * ly 1D vector
        return a lx * ly 1D vector
        '''
        lx, ly = b.shape[-1], b.shape[-2]
        return np.sqrt(lx * ly) * torch.real(
                    torch.fft.fft2(
                        torch.fft.fft2(b, norm="ortho") *
                        torch.fft.ifft2(v, norm="ortho"),
                    norm="ortho")
                )
    
    def get_base_invert(self, b):
        '''
        If b is the base of a matrix B, returns bi, the base of B^-1 with the
        direct formula sing Fourier space
        '''
        lx, ly = b.shape[-1], b.shape[-2]
        B = torch.fft.fft2(b, norm='ortho')
        res = 1 / (lx * ly) * torch.real(
            torch.fft.ifft2(
                torch.pow(B + 1e-6, -1),
                norm='ortho'
            )
        )
        return res

    def get_logdeterminant_base(self, covar_bases):
        '''
        Expects a [Batch, Channels, W, H], everything is vectorized over the
        first two channels
        '''
        lx, ly = covar_bases.shape[-1], covar_bases.shape[-2]
        B = np.sqrt(lx * ly) * torch.fft.fft2(covar_bases, norm="ortho")
        logdet = torch.sum(torch.log(
            (torch.real(B)) + 1e-6
            ), dim=(-2, -1))
        return logdet
