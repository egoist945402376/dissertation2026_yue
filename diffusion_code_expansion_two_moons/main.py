import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

from model import NoiseScheduler, ForwardProcess, BackwardProcess, DiffusionModel
from pac_bayes import compute_bound


def two_moons_data(num_samples, seed=None):
    """Sample a bounded two-moons distribution inside [-1, 1]^2."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, np.pi, num_samples)
    moon = rng.integers(0, 2, num_samples)

    x = np.where(moon == 0, np.cos(theta), 1 - np.cos(theta))
    y = np.where(moon == 0, np.sin(theta), 0.5 - np.sin(theta))

    # Centre and scale the classical moons, leaving room for bounded noise.
    X = 0.6 * np.stack((x - 0.5, y - 0.25), axis=1)
    X += rng.uniform(-0.05, 0.05, size=X.shape)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


if __name__ == '__main__':

    # the model
    ns = NoiseScheduler(timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule='linear')
    bp = BackwardProcess(hidden_layers=3, hidden_dim=128, embed_size=128, time_embed_type='sinusoidal', input_embed_type='sinusoidal')
    fp = ForwardProcess(noise_scheduler=ns)
    diff_model = DiffusionModel(forward_process=fp, backward_process=bp)

    # the data
    d = two_moons_data(num_samples=50000)
    # train_model() unpacks TensorDataset batches with ``batch = batch[0]``.
    # Passing ``d.tensors[0]`` here would incorrectly reduce each batch to
    # a single 2D point inside train_model().
    dl = DataLoader(d, batch_size=100, shuffle=True)

    # training
    diff_model.train_model(train_loader=dl, epochs=500, lr=1e-4)
    torch.save(diff_model.backward_process.state_dict(), 'two_moons_backward_process.pt')

    # compute bound
    bound_data = two_moons_data(num_samples=5000)
    bound_loader = DataLoader(bound_data.tensors[0], batch_size=100, shuffle=True)
    bound = compute_bound(data_loader=bound_loader, diff_model=diff_model, diameter=np.sqrt(8), lamda=5000, delta=0.05, dim=2)

    # show samples and originals
    plt.figure()
    real_samples = two_moons_data(num_samples=2000)
    plt.scatter(x=real_samples.tensors[0][:, 0], y=real_samples.tensors[0][:, 1], alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Real samples: two moons')
    plt.savefig('two_moons_real_samples.png')

    plt.figure()
    samples = diff_model.generate(2000, xlim=(-1, 1), ylim=(-1, 1))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Generated samples: two moons')
    plt.savefig('two_moons_fake_samples.png')

    print('Bound value: ', bound)
    plt.show()


