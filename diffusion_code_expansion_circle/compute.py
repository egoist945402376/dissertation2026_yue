import numpy as np

def checkerboard(n, rng):
    cells = np.array([(r, c) for r in range(4) for c in range(4) if (r + c) % 2 == 0])
    sel = cells[rng.integers(0, len(cells), n)]
    off = rng.uniform(0, 1, size=(n, 2))
    w = 0.5
    x = -1 + (sel[:, 1] + off[:, 0]) * w
    y = -1 + (sel[:, 0] + off[:, 1]) * w
    return np.stack((x, y), axis=1)

def two_moons(n, rng, noise=True):
    th = rng.uniform(0, np.pi, n)
    moon = rng.integers(0, 2, n)
    x = np.where(moon == 0, np.cos(th), 1 - np.cos(th))
    y = np.where(moon == 0, np.sin(th), 0.5 - np.sin(th))
    X = np.stack(((0.95/1.5)*(x-0.5), (0.95/0.75)*(y-0.25)), axis=1)
    if noise:
        X += rng.uniform(-0.05, 0.05, size=X.shape)
    return X

rng = np.random.default_rng(0)
m = 4_000_000



import numpy as np
rng = np.random.default_rng(0)
m = 2_000_000

# square
x = rng.uniform(-1, 1, (m, 2)); y = rng.uniform(-1, 1, (m, 2))
print("square  ", np.linalg.norm(x - y, axis=1).mean())   # 预期 ~1.043

# circle
t1 = rng.uniform(0, 2*np.pi, m); t2 = rng.uniform(0, 2*np.pi, m)
d = np.hypot(np.cos(t1)-np.cos(t2), np.sin(t1)-np.sin(t2))
print("circle  ", d.mean())                                # 预期 4/pi ≈ 1.273

for name, f in [("checkerboard", lambda n: checkerboard(n, rng)),
                ("two_moons (with noise, as coded)", lambda n: two_moons(n, rng, True)),
                ("two_moons (noise-free)", lambda n: two_moons(n, rng, False))]:
    a, b = f(m), f(m)
    d = np.linalg.norm(a - b, axis=1)
    print(f"{name:36s} {d.mean():.6f}  (se {d.std(ddof=1)/np.sqrt(m):.6f})")

a = two_moons(2_000_000, rng, True)
print("\ntwo_moons extent x:", a[:,0].min().round(4), a[:,0].max().round(4))
print("two_moons extent y:", a[:,1].min().round(4), a[:,1].max().round(4))