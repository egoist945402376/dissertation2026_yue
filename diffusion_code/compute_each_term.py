"""Compute and export every term of the released PAC-Bayes bound.

This script follows the implementation in ``pac_bayes.py`` exactly, including
the implementation-level differences from the formula in the paper.  The
lambda-independent quantities are computed once and reused for all six lambda
values, as in ``load_and_run.py``.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from main import rectangle_data
from model import BackwardProcess, DiffusionModel, ForwardProcess, NoiseScheduler
from pac_bayes import avg_distance, compute_last_term, empirical_risk, prior_matching


PAPER_BOUNDS = {
    "n/10": 1.124,
    "n/5": 1.231,
    "n/2": 1.5181,
    "n": 2.035,
    "n/0.5": 3.056,
    "n/0.1": 11.061,
}


def compute_lambda_independent_terms(data_loader, diff_model, dim):
    """Compute the four quantities reused for every lambda value.

    The expressions below intentionally mirror ``pac_bayes.compute_bound``.
    In particular, the released code uses ``sqrt(alpha_bar)`` in the
    Lipschitz proxy and ``1 - alpha`` as ``sigma_t``.
    """
    noise_sched = diff_model.forward_process.ns

    reconstruction = empirical_risk(data_loader, diff_model)
    prior_matching_sum = prior_matching(data_loader, noise_sched)

    lip_norm_forward = (
        torch.sqrt(noise_sched.alpha_bar)
        * (1 - noise_sched.alpha_bar_prev)
        / (1 - noise_sched.alpha_bar)
    )
    lip_norm_forward[0] = 1
    sigma_t = 1 - noise_sched.alpha

    average_terminal_distance = avg_distance(data_loader, diff_model)
    terminal_mismatch = average_terminal_distance * torch.prod(lip_norm_forward)
    accumulated_noise = compute_last_term(lip_norm_forward, sigma_t, dim)

    return {
        "reconstruction": reconstruction,
        "prior_matching_sum": prior_matching_sum,
        "average_terminal_distance": average_terminal_distance,
        "product_lipschitz": torch.prod(lip_norm_forward),
        "terminal_mismatch": terminal_mismatch,
        "accumulated_noise": accumulated_noise,
    }


def terms_for_lambda(fixed_terms, lamda, delta, diameter, n):
    """Return the released bound and its components for one lambda."""
    reconstruction = fixed_terms["reconstruction"]
    prior_matching_sum = fixed_terms["prior_matching_sum"]
    terminal_mismatch = fixed_terms["terminal_mismatch"]
    accumulated_noise = fixed_terms["accumulated_noise"]

    # This combined expression is exactly the second term in compute_bound().
    prior_matching_and_confidence = (
        prior_matching_sum + np.log(1 / delta)
    ) / lamda
    diameter_term = lamda * diameter**2 / (8 * n)

    total = (
        reconstruction
        + prior_matching_and_confidence
        + diameter_term
        + terminal_mismatch
        + accumulated_noise
    )

    # The split values below are included for interpretation.  Their sum is
    # exactly prior_matching_and_confidence, apart from display precision.
    prior_matching_over_lambda = prior_matching_sum / lamda
    confidence_over_lambda = np.log(1 / delta) / lamda

    return {
        "reconstruction": float(reconstruction),
        "prior_matching_sum_unscaled": float(prior_matching_sum),
        "prior_matching_over_lambda": float(prior_matching_over_lambda),
        "confidence_over_lambda": float(confidence_over_lambda),
        "prior_matching_and_confidence": float(prior_matching_and_confidence),
        "diameter_term": float(diameter_term),
        "average_terminal_distance": float(
            fixed_terms["average_terminal_distance"]
        ),
        "product_lipschitz": float(fixed_terms["product_lipschitz"]),
        "terminal_mismatch": float(terminal_mismatch),
        "accumulated_noise": float(accumulated_noise),
        "total_bound": float(total),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export every term of the released PAC-Bayes bound."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("backward_process.pt"),
        help="Path to the saved backward-process state dict.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bound_terms_by_lambda_1.csv"),
        help="Destination CSV path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ns = NoiseScheduler(
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",
    )
    bp = BackwardProcess(
        hidden_layers=3,
        hidden_dim=128,
        embed_size=128,
        time_embed_type="sinusoidal",
        input_embed_type="sinusoidal",
    )
    bp.load_state_dict(torch.load(args.model, map_location="cpu"))
    fp = ForwardProcess(noise_scheduler=ns)
    diff_model = DiffusionModel(forward_process=fp, backward_process=bp)

    bound_data = rectangle_data(num_samples=5000)
    bound_loader = DataLoader(
        bound_data.tensors[0], batch_size=100, shuffle=True
    )
    n = len(bound_loader.dataset)
    diameter = np.sqrt(8)
    delta = 0.05

    print("Computing lambda-independent terms (this is the slow part)...")
    fixed_terms = compute_lambda_independent_terms(
        bound_loader, diff_model, dim=2
    )

    lambdas = {
        "n/10": n / 10,
        "n/5": n / 5,
        "n/2": n / 2,
        "n": n,
        "n/0.5": n / 0.5,
        "n/0.1": n / 0.1,
    }

    rows = []
    print(
        f"\n{'lambda':<8}{'value':>9}{'recon':>11}{'prior+conf':>13}"
        f"{'diameter':>11}{'mismatch':>13}{'acc.noise':>13}"
        f"{'total':>11}{'paper':>10}{'diff':>10}"
    )
    print("-" * 109)

    for label, lamda in lambdas.items():
        terms = terms_for_lambda(
            fixed_terms, lamda, delta, diameter, n
        )
        paper_bound = PAPER_BOUNDS[label]
        difference = terms["total_bound"] - paper_bound
        row = {
            "lambda_label": label,
            "lambda_value": float(lamda),
            **terms,
            "paper_bound": paper_bound,
            "difference_from_paper": difference,
            "absolute_difference": abs(difference),
            "relative_difference_percent": (
                abs(difference) / paper_bound * 100
            ),
            "non_vacuous_below_sqrt8": terms["total_bound"] < diameter,
        }
        rows.append(row)

        print(
            f"{label:<8}{lamda:>9.0f}"
            f"{terms['reconstruction']:>11.6f}"
            f"{terms['prior_matching_and_confidence']:>13.6f}"
            f"{terms['diameter_term']:>11.6f}"
            f"{terms['terminal_mismatch']:>13.3e}"
            f"{terms['accumulated_noise']:>13.3e}"
            f"{terms['total_bound']:>11.6f}"
            f"{paper_bound:>10.4f}{difference:>+10.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
