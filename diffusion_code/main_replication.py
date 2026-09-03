from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

from model import NoiseScheduler, ForwardProcess, BackwardProcess, DiffusionModel
from pac_bayes import compute_bound


EXPERIMENT_DIR = Path(__file__).resolve().parent


def rectangle_data(num_samples):
    rng = np.random.default_rng()
    x = rng.uniform(-1, 1, num_samples)
    y = rng.uniform(-1, 1, num_samples)
    X = np.stack((x, y), axis=1)
    # X *= 4
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


if __name__ == '__main__':

    # the model
    ns = NoiseScheduler(timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule='linear')
    bp = BackwardProcess(hidden_layers=3, hidden_dim=128, embed_size=128, time_embed_type='sinusoidal', input_embed_type='sinusoidal')
    fp = ForwardProcess(noise_scheduler=ns)
    diff_model = DiffusionModel(forward_process=fp, backward_process=bp)

    # the data
    d = rectangle_data(num_samples=50000)
    dl = dataloader = DataLoader(d.tensors[0], batch_size=100, shuffle=True)

    # training
    diff_model.train_model(train_loader=dl, epochs=500, lr=1e-4)
    torch.save(
        diff_model.backward_process.state_dict(),
        EXPERIMENT_DIR / 'backward_process.pt',
    )

    # compute bound
    bound_data = rectangle_data(num_samples=5000)
    bound_loader = DataLoader(bound_data.tensors[0], batch_size=100, shuffle=True)
    bound = compute_bound(data_loader=bound_loader, diff_model=diff_model, diameter=np.sqrt(8), lamda=5000, delta=0.05, dim=2)

    # show samples and originals
    plt.figure()
    real_samples = rectangle_data(num_samples=2000)
    plt.scatter(x=real_samples.tensors[0][:, 0], y=real_samples.tensors[0][:, 1], alpha=0.5)
    plt.title('Real samples')
    plt.savefig(EXPERIMENT_DIR / 'real_samples.png')

    plt.figure()
    samples = diff_model.generate(2000, xlim=(-1, 1), ylim=(-1, 1))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
    plt.title('Fake samples')
    plt.savefig(EXPERIMENT_DIR / 'fake_samples.png')

    print('Bound value: ', bound)
    plt.show()



