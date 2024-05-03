import numpy as np
from tqdm import trange


def get_neighbour_matrix(x, L, R):
    dx = np.subtract.outer(x[:, 0], x[:, 0])
    dy = np.subtract.outer(x[:, 1], x[:, 1])
    dx[dx > (L / 2) ** 2] -= (L / 2) ** 2
    dy[dy > (L / 2) ** 2] -= (L / 2) ** 2
    pair_dist = dx ** 2 + dy ** 2
    neighbors = pair_dist < R ** 2
    return neighbors


def simulate_vicsek_model(
    N = 100,     # Number of individuals
    L = 10,      # Size of the domain
    R = 1,       # Interaction radius
    eta = 0.1,   # Noise level
    v = 1,       # Individual speed
    dt = 0.1,    # Time step
    T = 1000,    # Total timesteps
    ):

    # State variables
    x = np.nan * np.empty((T, N, 2))  # Position vectors
    e = np.nan * np.empty((T, N, 2))  # Orientation vectors

    # Initial conditions
    x[0] = L * np.random.uniform(0, L, (N, 2))
    theta = np.random.uniform(0, 2 * np.pi, N)
    e[0] = np.array([np.cos(theta), np.sin(theta)]).T

    # Main update loop
    for t in trange(1, T):
        neighbors = get_neighbour_matrix(x[t - 1], L, R)

        for i in range(N):
            # Compute the average orientation of the neighbors
            e[t, i] = e[t - 1, neighbors[i], :].mean(axis=0)

        # Add noise and normalize the orientation vectors
        e[t] += eta * np.random.normal(size=(N, 2))
        e[t] /= np.linalg.norm(e[t], axis=1)[:, None]

        # Update positions
        x[t] = np.mod(x[t - 1] + v * e[t] * dt, L)

    return x, e

if __name__ == '__main__':
    x, e = simulate_vicsek_model()
    print(x)
    print(e)