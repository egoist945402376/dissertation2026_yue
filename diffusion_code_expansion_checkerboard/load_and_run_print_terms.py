import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from main import checkerboard_data
from model import NoiseScheduler, ForwardProcess, BackwardProcess, DiffusionModel
from pac_bayes import empirical_risk, prior_matching, avg_distance, compute_last_term, expected_norm_diff_gaussian


def compute_bound_terms(data_loader, diff_model, dim):
    """Computes the parts of the bound that do NOT depend on lambda, once."""
    noise_sched = diff_model.forward_process.ns

    emp_risk = empirical_risk(data_loader, diff_model)
    prior_match = prior_matching(data_loader, noise_sched)

    # float64 throughout: the product of 1000 factors < 1 underflows in float32
    ab = noise_sched.alpha_bar.double()
    ab_prev = noise_sched.alpha_bar_prev.double()
    alpha = noise_sched.alpha.double()

    lip = torch.sqrt(alpha) * (1 - ab_prev) / (1 - ab)
    lip[0] = 1.0
    sigma_t = 1 - alpha                      # as in the released implementation

    # log-domain product, then exponentiate once
    log_prod_K = torch.log(lip).sum()
    prod_K = torch.exp(log_prod_K)

    avg_dist = avg_distance(data_loader, diff_model).double()
    avg_dist_term = avg_dist * prod_K
    last_term = compute_last_term_f64(lip, sigma_t, dim)

    diagnostics = {
        'alpha_bar_T': float(ab[-1]),
        'log_prod_K': float(log_prod_K),
        'prod_K_prime': float(prod_K),
        'avg_dist_raw': float(avg_dist),
    }
    return emp_risk.double(), prior_match.double(), avg_dist_term, last_term, diagnostics


def compute_last_term_f64(lip, sigma_t, dim):
    """float64 version of pac_bayes.compute_last_term, same summation as released."""
    T = len(lip)
    cum = torch.cumprod(lip, dim=0)          # cum[k] = prod_{i=1}^{k+1} K_i
    total = torch.zeros((), dtype=torch.float64)
    for t in range(2, T + 1):
        # released code multiplies sigma_t inside the inner loop, i.e. sigma_t**(t-1)
        total = total + cum[t - 2] * sigma_t[t - 1] ** (t - 1)
    e_norm = expected_norm_diff_gaussian(num_samples=int(1e6), dim=dim)
    return total * float(e_norm)


def bound_for_lambda(emp_risk, prior_match, avg_dist_term, last_term, lamda, delta, diameter, n):
    """Returns the total bound together with each individual term."""
    kl_term = (prior_match + np.log(1 / delta)) / lamda
    diam_term = lamda * diameter ** 2 / (8 * n)
    total = emp_risk + kl_term + diam_term + avg_dist_term + last_term
    return {
        'reconstruction': float(emp_risk),
        'kl_over_lambda': float(kl_term),
        'diameter': float(diam_term),
        'terminal_mismatch': float(avg_dist_term),
        'accumulated_noise': float(last_term),
        'C0': float(emp_risk + avg_dist_term + last_term),
        'total': float(total),
    }


if __name__ == '__main__':

    ns = NoiseScheduler(timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule='linear')
    bp = BackwardProcess(hidden_layers=3, hidden_dim=128, embed_size=128,
                         time_embed_type='sinusoidal', input_embed_type='sinusoidal')
    bp.load_state_dict(torch.load('checkerboard_backward_process.pt', map_location='cpu'))
    fp = ForwardProcess(noise_scheduler=ns)
    diff_model = DiffusionModel(forward_process=fp, backward_process=bp)

    bound_data = checkerboard_data(num_samples=5000)
    bound_loader = DataLoader(bound_data.tensors[0], batch_size=100, shuffle=True)
    n = len(bound_loader.dataset)
    diameter = np.sqrt(8)
    delta = 0.05

    print('Dataset: 4x4 checkerboard in [-1, 1]^2; certificate diameter: sqrt(8)')
    print('Computing the lambda-independent terms (this is the slow part, run once)...')
    emp_risk, prior_match, avg_dist_term, last_term, diag = compute_bound_terms(bound_loader, diff_model, dim=2)

    print('\n=== Fixed by the noise schedule (independent of training) ===')
    print(f"  alpha_bar_T        = {diag['alpha_bar_T']:.6e}")
    print(f"  prod_t K'_t        = {diag['prod_K_prime']:.6e}")

    print('\n=== Lambda-independent terms ===')
    print(f"  reconstruction     = {float(emp_risk):.6f}")
    print(f"  KL sum (unscaled)  = {float(prior_match):.6f}")
    print(f"  terminal mismatch  = {float(avg_dist_term):.6e}   "
          f"(= {diag['avg_dist_raw']:.4f} x prod K'_t)")
    print(f"  accumulated noise  = {float(last_term):.6e}")
    print(f"  C0                 = {float(emp_risk + avg_dist_term + last_term):.6f}")

    lambdas = {'n/10': n / 10, 'n/5': n / 5, 'n/2': n / 2,
               'n': n, 'n/0.5': n / 0.5, 'n/0.1': n / 0.1}
    header = (f"\n{'lambda':<8}{'value':>8}{'recon':>10}{'KL/lam':>9}{'diam':>9}"
              f"{'mismatch':>12}{'acc.noise':>12}{'C0':>10}{'total':>10}")
    print(header)
    print('-' * (len(header) - 1))

    rows = []
    for label, lamda in lambdas.items():
        t = bound_for_lambda(emp_risk, prior_match, avg_dist_term, last_term,
                             lamda, delta, diameter, n)
        print(f"{label:<8}{lamda:>8.0f}{t['reconstruction']:>10.4f}{t['kl_over_lambda']:>9.4f}"
              f"{t['diameter']:>9.4f}{t['terminal_mismatch']:>12.2e}{t['accumulated_noise']:>12.2e}"
              f"{t['C0']:>10.4f}{t['total']:>10.4f}")
        rows.append({'dataset': 'checkerboard', 'lambda_label': label,
                     'lambda': lamda, **t})

    print('\n=== Term shares (% of total) ===')
    for label in ['n/10', 'n', 'n/0.1']:
        r = next(x for x in rows if x['lambda_label'] == label)
        print(f"  lambda = {label:<6} recon {100*r['reconstruction']/r['total']:5.1f}%   "
              f"diam {100*r['diameter']/r['total']:5.1f}%   "
              f"KL/lam {100*r['kl_over_lambda']/r['total']:5.2f}%   "
              f"mismatch+noise {100*(r['terminal_mismatch']+r['accumulated_noise'])/r['total']:.3f}%")

    print(f"\nNon-vacuity baseline: Delta = {diameter:.4f}")
    for r in rows:
        flag = 'non-vacuous' if r['total'] < diameter else 'VACUOUS'
        print(f"  {r['lambda_label']:<8}{r['total']:>10.4f}   {flag}")

    with open('checkerboard_bound_terms.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('\nWritten to checkerboard_bound_terms.csv')
