import torch
import numpy as np


def compute_bound(data_loader, diff_model, diameter, lamda, delta, dim):
    n = len(data_loader.dataset)
    noise_sched = diff_model.forward_process.ns

    emp_risk = empirical_risk(data_loader, diff_model)
    prior_match = prior_matching(data_loader, noise_sched)
    diam_term = lamda * diameter**2 / (8 * n)
    lip_norm_forward = torch.sqrt(noise_sched.alpha_bar) * (1 - noise_sched.alpha_bar_prev) / (1 - noise_sched.alpha_bar)  # K_t' from remark 3.2
    lip_norm_forward[0] = 1  # to account for the padding
    sigma_t = 1 - noise_sched.alpha  # same as the variance for the forward.
    avg_dist = avg_distance(data_loader, diff_model)
    avg_dist_term = avg_dist * torch.prod(lip_norm_forward)
    last_term = compute_last_term(lip_norm_forward, sigma_t, dim)

    return emp_risk + (prior_match + np.log(1/delta)) / lamda + diam_term + avg_dist_term + last_term


def empirical_risk(data_loader, diff_model):
    noise_sched = diff_model.forward_process.ns
    emp_risk_sum = torch.Tensor([0.])
    for batch in data_loader:
        num_samples = batch.shape[0]
        samples = torch.sqrt(noise_sched.alpha_bar[-1]) * batch + torch.sqrt(1 - noise_sched.alpha_bar[-1]) * torch.randn_like(batch)
        timesteps_reverse = list(range(noise_sched.timesteps))[::-1]
        for t in timesteps_reverse:
            t = torch.Tensor([t] * num_samples).long()
            with torch.no_grad():
                pred_noise = diff_model.backward_process(samples, t)
                samples = diff_model.forward_process.get_x_t_min_one(x_t=samples, t=t[0], noise=pred_noise)
        # End of the backwards process
        emp_risk_sum += torch.sum(torch.norm(batch - samples, p=2, dim=1))
    return emp_risk_sum / len(data_loader.dataset)


def prior_matching(data_loader, noise_sched):
    kl_sum = torch.Tensor([0.])
    for batch in data_loader:
        mu = torch.sqrt(noise_sched.alpha_bar[-1]) * batch
        sigma_sq = 1 - noise_sched.alpha_bar[-1]
        kl_sum += kl_divergence_sum(mu=mu, sigma_sq=sigma_sq)
    return kl_sum


def kl_divergence_sum(mu, sigma_sq):
    return -0.5 * torch.sum(1 + sigma_sq.log() - mu ** 2 - sigma_sq)


def compute_last_term(lip_norm_forward, sigma_t, dim):
    timesteps = len(lip_norm_forward)
    result = 0
    for t in range(2, timesteps + 1):
        prod = 1
        for i in range(1, t):
            prod *= lip_norm_forward[i-1] * sigma_t[t-1]
        result += prod
    result *= expected_norm_diff_gaussian(num_samples=int(1e6), dim=dim)
    return result


def avg_distance(data_loader, diff_model, num_samples=int(1e6)):
    avg_distance_sum = torch.Tensor([0.])
    for batch in data_loader:
        for i in range(batch.shape[0]):
            x = batch[i]
            mean = torch.sqrt(diff_model.forward_process.ns.alpha_bar[-1]) * x
            sigma_sq = 1 - diff_model.forward_process.ns.alpha_bar[-1]
            avg_distance_sum += expected_norm_diff_gaussian(num_samples=num_samples, dim=batch.shape[1],
                                                            mu_1=mean, sigma_sq_1=sigma_sq, mu_2=None, sigma_sq_2=None)
    return avg_distance_sum / len(data_loader.dataset)


def expected_norm_diff_gaussian(num_samples, dim, mu_1=None, sigma_sq_1=None, mu_2=None, sigma_sq_2=None):
    # For each distribution, if the parameters are None, N(0, I) is considered
    eps_1 = torch.randn(num_samples, dim)
    eps_2 = torch.randn(num_samples, dim)
    mu_1 = torch.zeros(num_samples, dim) if mu_1 is None else mu_1
    mu_2 = torch.zeros(num_samples, dim) if mu_2 is None else mu_2
    sigma_sq_1 = torch.ones(num_samples, dim) if sigma_sq_1 is None else sigma_sq_1
    sigma_sq_2 = torch.ones(num_samples, dim) if sigma_sq_2 is None else sigma_sq_2

    x_1 = mu_1 + torch.sqrt(sigma_sq_1) * eps_1
    x_2 = mu_2 + torch.sqrt(sigma_sq_2) * eps_2

    return torch.mean(torch.norm(input=x_1 - x_2, dim=1, p=2))





