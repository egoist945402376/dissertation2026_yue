import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

from model import NoiseScheduler, ForwardProcess, BackwardProcess, DiffusionModel
from pac_bayes import compute_bound


def checkerboard_data(num_samples, seed=None):
    """Sample uniformly from alternating cells of a 4x4 checkerboard."""
    rng = np.random.default_rng(seed)
    cells = np.array(
        [(row, col) for row in range(4) for col in range(4)
         if (row + col) % 2 == 0],
        dtype=np.int64,
    )
    selected = cells[rng.integers(0, len(cells), num_samples)]
    offsets = rng.uniform(0, 1, size=(num_samples, 2))
    cell_width = 0.5
    x = -1 + (selected[:, 1] + offsets[:, 0]) * cell_width
    y = -1 + (selected[:, 0] + offsets[:, 1]) * cell_width
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


if __name__ == '__main__':

    # the model
    ns = NoiseScheduler(timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule='linear')
    bp = BackwardProcess(hidden_layers=3, hidden_dim=128, embed_size=128, time_embed_type='sinusoidal', input_embed_type='sinusoidal')
    fp = ForwardProcess(noise_scheduler=ns)
    diff_model = DiffusionModel(forward_process=fp, backward_process=bp)

    # the data
    d = checkerboard_data(num_samples=50000)
    dl = dataloader = DataLoader(d.tensors[0], batch_size=100, shuffle=True)

    # training
    diff_model.train_model(train_loader=dl, epochs=500, lr=1e-4)
    torch.save(diff_model.backward_process.state_dict(), 'checkerboard_backward_process.pt')

    # compute bound
    bound_data = checkerboard_data(num_samples=5000)
    bound_loader = DataLoader(bound_data.tensors[0], batch_size=100, shuffle=True)
    bound = compute_bound(data_loader=bound_loader, diff_model=diff_model, diameter=np.sqrt(8), lamda=5000, delta=0.05, dim=2)

    # show samples and originals
    plt.figure()
    real_samples = checkerboard_data(num_samples=2000)
    plt.scatter(x=real_samples.tensors[0][:, 0], y=real_samples.tensors[0][:, 1], alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Real samples: checkerboard')
    plt.savefig('checkerboard_real_samples.png')

    plt.figure()
    samples = diff_model.generate(2000, xlim=(-1, 1), ylim=(-1, 1))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Generated samples: checkerboard')
    plt.savefig('checkerboard_fake_samples.png')

    print('Bound value: ', bound)
    plt.show()



