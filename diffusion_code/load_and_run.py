import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from main import rectangle_data
from model import NoiseScheduler, ForwardProcess, BackwardProcess, DiffusionModel
from pac_bayes import compute_bound


if __name__ == '__main__':
    # rebuild the exact same architecture used in main.py, then load the trained weights
    ns = NoiseScheduler(timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule='linear')
    bp = BackwardProcess(hidden_layers=3, hidden_dim=128, embed_size=128, time_embed_type='sinusoidal', input_embed_type='sinusoidal')
    bp.load_state_dict(torch.load('backward_process.pt'))
    fp = ForwardProcess(noise_scheduler=ns)
    diff_model = DiffusionModel(forward_process=fp, backward_process=bp)

    # bound
    bound_data = rectangle_data(num_samples=5000)
    bound_loader = DataLoader(bound_data.tensors[0], batch_size=100, shuffle=True)
    bound = compute_bound(data_loader=bound_loader, diff_model=diff_model, diameter=np.sqrt(8), lamda=5000, delta=0.05, dim=2)
    print('Bound value: ', bound)

    # plots
    plt.figure()
    real_samples = rectangle_data(num_samples=2000)
    plt.scatter(x=real_samples.tensors[0][:, 0], y=real_samples.tensors[0][:, 1], alpha=0.5)
    plt.title('Real samples')
    plt.savefig('real_samples.png')

    plt.figure()
    samples = diff_model.generate(2000, xlim=(-1, 1), ylim=(-1, 1))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
    plt.title('Fake samples')
    plt.savefig('fake_samples.png')

    plt.show()
