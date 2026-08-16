import numpy as np
import torch
from torch.utils.data import DataLoader

from main import rectangle_data
from model import NoiseScheduler, ForwardProcess, BackwardProcess, DiffusionModel
from pac_bayes import empirical_risk, prior_matching, avg_distance, compute_last_term


def compute_bound_terms(data_loader, diff_model, dim):
    """Computes the parts of the bound that do NOT depend on lambda, once."""
    noise_sched = diff_model.forward_process.ns

    emp_risk = empirical_risk(data_loader, diff_model)
    prior_match = prior_matching(data_loader, noise_sched)

    lip_norm_forward = torch.sqrt(noise_sched.alpha_bar) * (1 - noise_sched.alpha_bar_prev) / (1 - noise_sched.alpha_bar)
    lip_norm_forward[0] = 1
    sigma_t = 1 - noise_sched.alpha

    avg_dist = avg_distance(data_loader, diff_model)
    avg_dist_term = avg_dist * torch.prod(lip_norm_forward)
    last_term = compute_last_term(lip_norm_forward, sigma_t, dim)

    return emp_risk, prior_match, avg_dist_term, last_term


def bound_for_lambda(emp_risk, prior_match, avg_dist_term, last_term, lamda, delta, diameter, n):
    diam_term = lamda * diameter ** 2 / (8 * n)
    return emp_risk + (prior_match + np.log(1 / delta)) / lamda + diam_term + avg_dist_term + last_term


if __name__ == '__main__':
    torch.manual_seed(0)

    # the model (same setup as main.py / paper Appendix B)
    ns = NoiseScheduler(timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule='linear')
    bp = BackwardProcess(hidden_layers=3, hidden_dim=128, embed_size=128, time_embed_type='sinusoidal', input_embed_type='sinusoidal')
    fp = ForwardProcess(noise_scheduler=ns)
    diff_model = DiffusionModel(forward_process=fp, backward_process=bp)

    # training data: 50,000 samples, as in the paper
    d = rectangle_data(num_samples=50000)
    dl = DataLoader(d.tensors[0], batch_size=100, shuffle=True)
    diff_model.train_model(train_loader=dl, epochs=500, lr=1e-4)

    # bound data: n = 5,000 samples, as in the paper
    bound_data = rectangle_data(num_samples=5000)
    bound_loader = DataLoader(bound_data.tensors[0], batch_size=100, shuffle=True)
    n = len(bound_loader.dataset)

    print('Computing the lambda-independent terms (this is the slow part, run once)...')
    emp_risk, prior_match, avg_dist_term, last_term = compute_bound_terms(bound_loader, diff_model, dim=2)

    diameter = np.sqrt(8)
    delta = 0.05
    lambdas = {'n/10': n / 10, 'n/5': n / 5, 'n/2': n / 2, 'n': n, 'n/0.5': n / 0.5, 'n/0.1': n / 0.1}

    print('\nlambda\tvalue\tbound')
    results = {}
    for label, lamda in lambdas.items():
        bound = bound_for_lambda(emp_risk, prior_match, avg_dist_term, last_term, lamda, delta, diameter, n)
        results[label] = float(bound)
        print(f'{label}\t{lamda:.1f}\t{float(bound):.3f}')

    print('\nPaper (Table, page 12): n/10=1.124  n/5=1.231  n/2=1.518  n=2.035  n/0.5=3.056  n/0.1=11.061')
