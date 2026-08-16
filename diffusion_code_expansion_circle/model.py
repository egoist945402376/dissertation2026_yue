import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from positional_embeddings import PositionalEmbedding


class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule='linear'):
        self.timesteps = timesteps  # this is T

        # betas
        if beta_schedule == 'linear':
            self.beta = torch.linspace(start=beta_start, end=beta_end, steps=timesteps, dtype=torch.float32)
        elif beta_schedule == 'quadratic':
            self.beta = torch.square(torch.linspace(start=np.sqrt(beta_start), end=np.sqrt(beta_end), steps=timesteps, dtype=torch.float32))

        # alphas et \bar{alpha}
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar_prev = F.pad(input=self.alpha_bar[:-1], pad=(1, 0), value=1.)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.inv_alpha_bar = 1 / self.alpha_bar

        # This is for a test to see what K_t roughly looks like
        self.muq_lip = torch.sqrt(self.alpha) * (1-self.alpha_bar_prev) / (1-self.alpha_bar)

        # required for mu_q
        self.muq_coeff_x0 = (1 - self.alpha) * torch.sqrt(self.alpha_bar_prev) / (1 - self.alpha_bar)
        self.muq_coeff_xt = (1 - self.alpha_bar_prev) * torch.sqrt(self.alpha) / (1 - self.alpha_bar)


class BackwardProcess(nn.Module):
    def __init__(self, hidden_layers=3, hidden_dim=100, embed_size=128, time_embed_type='sinusoidal', input_embed_type='sinusoidal'):
        super(BackwardProcess, self).__init__()

        # The embeddings
        self.time_embed = PositionalEmbedding(size=embed_size, type=time_embed_type)
        self.input_embed_1 = PositionalEmbedding(size=embed_size, type=input_embed_type)
        self.input_embed_2 = PositionalEmbedding(size=embed_size, type=input_embed_type)

        # The layers
        concat_size = len(self.time_embed.layer) + len(self.input_embed_1.layer) + len(self.input_embed_2.layer)
        layers = [nn.Linear(in_features=concat_size, out_features=hidden_dim), nn.GELU()]
        for _ in range(hidden_layers):
            layers += [nn.Linear(in_features=hidden_dim, out_features=hidden_dim), nn.GELU()]
        layers += [nn.Linear(in_features=hidden_dim, out_features=2)]
        self.network = nn.Sequential(*layers)

    def forward(self, x_t, t):
        # Compute the embeddings
        time_embedded = self.time_embed(t)
        x1_embedded = self.input_embed_1(x_t[:, 0])
        x2_embedded = self.input_embed_2(x_t[:, 1])

        x = torch.cat([x1_embedded, x2_embedded, time_embedded], dim=-1)
        return self.network(x)


class ForwardProcess:
    def __init__(self, noise_scheduler: NoiseScheduler):
        self.ns = noise_scheduler

    def get_x0(self, x_t, t, noise):
        coeff_xt = 1 / self.ns.sqrt_alpha_bar[t]
        coeff_noise = torch.sqrt(self.ns.inv_alpha_bar - 1)[t]
        coeff_xt, coeff_noise = coeff_xt.reshape(-1, 1), coeff_noise.reshape(-1, 1)
        return coeff_xt * x_t - coeff_noise * noise  # it is a minus, not a plus

    def get_mu_q(self, x_0, x_t, t):
        coeff_x0 = self.ns.muq_coeff_x0[t]
        coeff_xt = self.ns.muq_coeff_xt[t]
        coeff_x0, coeff_xt = coeff_x0.reshape(-1, 1), coeff_xt.reshape(-1, 1)
        return coeff_x0 * x_0 + coeff_xt * x_t

    def get_sigma_q(self, t):
        variance = (1 - self.ns.alpha[t]) * (1 - self.ns.alpha_bar_prev[t]) / (1-self.ns.alpha_bar[t])
        variance = variance.clip(1e-20)
        return torch.Tensor([0]) if t == 0 else variance

    def get_x_t_min_one(self, x_t, t, noise):
        pred_x0 = self.get_x0(x_t=x_t, t=t, noise=noise)
        mu_q = self.get_mu_q(x_0=pred_x0, x_t=x_t, t=t)

        variance = torch.randn_like(noise) * torch.sqrt(self.get_sigma_q(t))
        return mu_q + variance

    def get_noisy_samples(self, x_0, noise, timesteps):
        coeffs_x0 = self.ns.sqrt_alpha_bar[timesteps]
        coeffs_noise = torch.sqrt(1 - self.ns.alpha_bar[timesteps])
        coeffs_x0, coeffs_noise = coeffs_x0.reshape(-1, 1), coeffs_noise.reshape(-1, 1)
        return coeffs_x0 * x_0 + coeffs_noise * noise


class DiffusionModel:
    def __init__(self, backward_process: BackwardProcess, forward_process: ForwardProcess):
        self.backward_process = backward_process
        self.forward_process = forward_process

    def train_model(self, train_loader, epochs, lr=1e-4):
        self.backward_process.train()
        self.backward_process = self.backward_process
        optimizer = torch.optim.AdamW(params=self.backward_process.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            print('Epoch {} ...'.format(epoch))
            losses = []
            for batch in tqdm(train_loader):
                batch = batch[0]
                noise = torch.randn_like(batch)

                timesteps = torch.randint(low=0, high=self.forward_process.ns.timesteps, size=(batch.shape[0],)).long()
                noisy_samples = self.forward_process.get_noisy_samples(x_0=batch, noise=noise, timesteps=timesteps)
                noise_pred = self.backward_process(x_t=noisy_samples, t=timesteps)
                loss = F.mse_loss(input=noise_pred, target=noise)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(float(loss.detach()))

            # End of Epoch
            avg_loss = np.mean(losses)
            print('Loss: {:.3f}'.format(avg_loss))

    def generate(self, num_samples, xlim=None, ylim=None):
        print('Generating samples ...')
        self.backward_process.eval()
        samples = torch.randn(num_samples, 2)
        timesteps_reverse = list(range(1, self.forward_process.ns.timesteps))[::-1]  # From t=T-1 to t=1
        for t in tqdm(timesteps_reverse):
            t = torch.Tensor([t] * num_samples).long()
            with torch.no_grad():
                pred_noise = self.backward_process(samples, t)
                samples = self.forward_process.get_x_t_min_one(x_t=samples, t=t[0], noise=pred_noise)

        # t=0 Clamp the coordinates, if needed
        if xlim is not None and ylim is not None:
            samples[:, 0] = torch.clamp(samples[:, 0], min=xlim[0], max=xlim[1])
            samples[:, 1] = torch.clamp(samples[:, 1], min=ylim[0], max=ylim[1])

        return samples

